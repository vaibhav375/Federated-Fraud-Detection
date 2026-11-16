"""Every Stage A0 baseline and privacy setting on the indexed IBM AML dataset.

All models share one chronological split. Model selection uses pooled
validation PR-AUC; test data is scored once, for the selected weights.

    python3 experiments/run_ibm_comparison.py [--max-train-accounts N] [--seeds 0 1 2]

Results (JSON): one entry per model, per seed: pooled and per-client test
metrics, privacy spent (DP models), and training time.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_training import (DEFAULT_CONFIG, DifferentialPrivacy, add_server_noise,  # noqa: E402
                          federated_aggregate, train_local)
from src.data.sequences import load_indexed_client  # noqa: E402
from src.federated.dp_accounting import DPFedAvgAccountant  # noqa: E402
from src.federated.dp_random import make_dp_random  # noqa: E402
from src.federated.dp_sgd import (AccountDPConfig, freeze_non_dp_heads,  # noqa: E402
                                  noise_multiplier_for, train_local_account_dp)
from src.models.temporal_graph_transformer import TemporalGraphTransformer  # noqa: E402
from src.utils.metrics import best_f1_threshold, fraud_metrics  # noqa: E402

ALL_MODELS = ["local", "centralized", "gbdt", "fedavg", "fedprox", "client_dp", "account_dp"]
SCORE_BATCH = 4096


# ---------------------------------------------------------------- data access

class BatchIter:
    """Batches of (seqs, labels) from a SequenceIndexDataset, optionally drawing
    fraud at a fixed rate (the oversampling the v1 pipeline found useful)."""

    def __init__(self, ds, batch_size, oversample_ratio=None, generator=None):
        self.ds, self.batch_size, self.ratio, self.g = ds, batch_size, oversample_ratio, generator
        y = ds.example_labels
        self.pos = (y == 1).nonzero(as_tuple=True)[0]
        self.neg = (y == 0).nonzero(as_tuple=True)[0]

    def __len__(self):
        return math.ceil(len(self.ds) / self.batch_size)

    def __iter__(self):
        n = len(self.ds)
        if self.ratio and len(self.pos) and len(self.neg):
            is_pos = torch.rand(n, generator=self.g) < self.ratio
            idx = torch.where(is_pos,
                              self.pos[torch.randint(len(self.pos), (n,), generator=self.g)],
                              self.neg[torch.randint(len(self.neg), (n,), generator=self.g)])
        else:
            idx = torch.randperm(n, generator=self.g)
        for s in range(0, n, self.batch_size):
            yield self.ds.gather(idx[s:s + self.batch_size])


def score(model, ds) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ds), SCORE_BATCH):
            seqs, _ = ds.gather(torch.arange(s, min(s + SCORE_BATCH, len(ds))))
            out.append(model(seqs, labels=None)["fraud_prob"].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def labels_of(ds) -> np.ndarray:
    return ds.example_labels.numpy()


# ---------------------------------------------------------------- evaluation

def val_pr_auc(per_client_probs, clients) -> float:
    y = np.concatenate([labels_of(clients[c]["val"]) for c in clients])
    p = np.concatenate([per_client_probs[c] for c in clients])
    return fraud_metrics(y, p)["pr_auc"] or -1.0


def evaluate_split(models_by_client, clients, split):
    return {c: score(models_by_client[c], clients[c][split]) for c in clients}


def final_metrics(val_probs, test_probs, clients) -> dict:
    """Threshold from pooled validation scores; metrics on pooled and per-client test."""
    yv = np.concatenate([labels_of(clients[c]["val"]) for c in clients])
    thr = best_f1_threshold(yv, np.concatenate([val_probs[c] for c in clients]))
    yt = np.concatenate([labels_of(clients[c]["test"]) for c in clients])
    pt = np.concatenate([test_probs[c] for c in clients])
    return {"pooled": fraud_metrics(yt, pt, thr),
            "per_client": {str(c): fraud_metrics(labels_of(clients[c]["test"]), test_probs[c], thr)
                           for c in clients}}


def select_and_test(states, clients, build_model) -> dict:
    """states: list of (val_score, {client: state_dict}) - pick best by val, test once."""
    _, best = max(states, key=lambda s: s[0])
    models = {}
    for c in clients:
        m = build_model()
        m.load_state_dict(best[c])
        models[c] = m
    return final_metrics(evaluate_split(models, clients, "val"), evaluate_split(models, clients, "test"), clients)


# ---------------------------------------------------------------- models

def model_builder(config):
    """Builder for the deep model with the zero-weight heads frozen.

    The contrastive and auxiliary losses have weight 0, but the contrastive
    temperature is a learnable parameter: Adam's weight decay alone walks it to
    0, sim/T then overflows and 0 * inf = NaN reaches every gradient (observed at
    batch 788 of one centralized epoch). Freezing the unused heads removes that."""
    def build():
        m = TemporalGraphTransformer(config)
        freeze_non_dp_heads(m)
        return m
    return build


def aggregate(global_model, locals_, sizes, build_model):
    """Weighted average of local models, keeping the global contrastive buffers.

    train_local fills each local memory bank with projections of real training
    rows, and the DP mechanisms clip and noise parameters only, so averaging the
    buffers would release that bank outside the privacy guarantee."""
    for m in locals_:
        if hasattr(m, "contrastive_head"):  # a head-only model has no buffers to protect
            m.contrastive_head.load_state_dict(global_model.contrastive_head.state_dict())
    return federated_aggregate(build_model(), locals_, sizes)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def is_finite(model) -> bool:
    return all(torch.isfinite(p).all() for p in model.parameters())


def run_local(args, clients, build_model, g):
    """One model per client, own data only; each picks its own best epoch on its own val."""
    best_states = {}
    for c in clients:
        m = build_model()
        opt = torch.optim.Adam(trainable(m), lr=args.lr, weight_decay=1e-5)
        best_score, best_states[c] = -2.0, snapshot(m)
        for _ in range(args.rounds * args.local_epochs):
            train_local(m, BatchIter(clients[c]["train"], args.batch_size, args.oversample_ratio, g),
                        opt, None, None, "cpu", global_params=None, mu=0.0, epochs=1)
            pv = score(m, clients[c]["val"])
            s = fraud_metrics(labels_of(clients[c]["val"]), pv)["pr_auc"] or -1.0
            if s > best_score and is_finite(m):
                best_score, best_states[c] = s, snapshot(m)
    return select_and_test([(0.0, best_states)], clients, build_model), None


class _Pooled:
    """All clients' train examples as one dataset (centralized baseline)."""

    def __init__(self, datasets):
        self.parts = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets])
        self.example_labels = torch.cat([d.example_labels for d in datasets])

    def __len__(self):
        return int(self.offsets[-1])

    def gather(self, idx):
        idx = idx.numpy()
        part = np.searchsorted(self.offsets, idx, side="right") - 1
        seqs = labels = None
        for k in np.unique(part):
            sel = np.nonzero(part == k)[0]
            s, l = self.parts[k].gather(torch.from_numpy(idx[sel] - self.offsets[k]))
            if seqs is None:
                seqs = torch.empty((len(idx),) + s.shape[1:])
                labels = torch.empty(len(idx), dtype=l.dtype)
            seqs[sel], labels[sel] = s, l
        return seqs, labels


def run_centralized(args, clients, build_model, g):
    m = build_model()
    opt = torch.optim.Adam(trainable(m), lr=args.lr, weight_decay=1e-5)
    pooled = _Pooled([clients[c]["train"] for c in clients])
    states = []
    for _ in range(args.rounds * args.local_epochs):
        train_local(m, BatchIter(pooled, args.batch_size, args.oversample_ratio, g),
                    opt, None, None, "cpu", global_params=None, mu=0.0, epochs=1)
        if is_finite(m):
            states.append((val_pr_auc(evaluate_split({c: m for c in clients}, clients, "val"), clients),
                           {c: snapshot(m) for c in clients}))
    return select_and_test(states or [(0.0, {c: snapshot(m) for c in clients})], clients, build_model), None


def gbdt_features(ds) -> np.ndarray:
    seqs, _ = ds.gather(torch.arange(len(ds)))
    from src.data.ibm_aml import PAD_COL
    real = (seqs[..., PAD_COL] == 0).float().unsqueeze(-1)
    mean = (seqs * real).sum(1) / real.sum(1).clamp(min=1)
    return torch.cat([seqs[:, -1], mean], dim=1).numpy()


def run_gbdt(args, clients, seed):
    X = np.concatenate([gbdt_features(clients[c]["train"]) for c in clients])
    y = np.concatenate([labels_of(clients[c]["train"]) for c in clients])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                         early_stopping=False, random_state=seed).fit(X, y)
    probs = {s: {c: clf.predict_proba(gbdt_features(clients[c][s]))[:, 1] for c in clients}
             for s in ("val", "test")}
    return final_metrics(probs["val"], probs["test"], clients), None


def account_dp_plan(args, n_accounts: int, eps: float) -> dict:
    """Per-bank sampling rate, steps per round and noise multiplier for account-level DP."""
    q = min(1.0, args.dp_batch_accounts / n_accounts)
    steps = args.local_epochs * math.ceil(1 / q)
    return {"sample_rate": q, "steps_per_round": steps, "n_accounts": n_accounts,
            "sigma": noise_multiplier_for(eps, args.delta, q, args.rounds * steps)}


def local_fit(model, client, args, global_params, generator, *, mu, account_dp=None, client_dp=None):
    """One round of local training in place. account_dp = (AccountDPConfig, plan, accountant)."""
    opt = torch.optim.Adam(trainable(model), lr=args.lr,
                           weight_decay=0.0 if account_dp is not None else 1e-5)
    if account_dp is not None:
        cfg, d, accountant = account_dp
        train_local_account_dp(model, client["train"], opt, cfg, d["sigma"], d["sample_rate"],
                               d["steps_per_round"], accountant, global_params=global_params,
                               mu=mu, pos_weight=args.pos_weight, generator=generator,
                               dp_rng=make_dp_random(args, generator))
    else:
        train_local(model, BatchIter(client["train"], args.batch_size, args.oversample_ratio, generator),
                    opt, client_dp, None, "cpu", global_params=global_params, mu=mu,
                    epochs=args.local_epochs)


def run_federated(args, clients, build_model, g, *, mu, client_dp_eps=None, account_dp_eps=None,
                  include_initial=False):
    """FedAvg/FedProx with all clients each round; optional client-level or account-level DP.

    include_initial: the starting model is scored on validation as round 0 and can be selected,
    so fine-tuning a pretrained model is only kept when it helps."""
    global_model = build_model()
    sizes = {c: len(clients[c]["train"]) for c in clients}
    privacy, dp_plan, client_acc = None, {}, None
    if account_dp_eps is not None:
        for c in clients:
            dp_plan[c] = {**account_dp_plan(args, clients[c]["n_train_accounts"], account_dp_eps),
                          "accountant": RDPAccountant()}
    if client_dp_eps is not None:
        client_acc = DPFedAvgAccountant(delta=args.delta)
    dp_cfg = AccountDPConfig(account_dp_eps or 0.0, args.delta, args.clip_norm,
                             args.dp_batch_accounts, args.dp_examples_per_account,
                             fraud_rows_first=getattr(args, "dp_fraud_rows_first", False))

    def state_of(model):
        return (val_pr_auc(evaluate_split({c: model for c in clients}, clients, "val"), clients),
                {c: snapshot(model) for c in clients})

    states = [state_of(global_model)] if include_initial else []
    for _ in range(args.rounds):
        prev = snapshot(global_model)
        global_params = [p.detach().clone() for p in global_model.parameters()]
        dp = (DifferentialPrivacy(max_norm=args.clip_norm, epsilon=client_dp_eps, delta=args.delta)
              if client_dp_eps is not None else None)
        locals_ = []
        for c in clients:
            m = build_model()
            m.load_state_dict(global_model.state_dict())
            local_fit(m, clients[c], args, global_params, g, mu=mu, client_dp=dp,
                      account_dp=(dp_cfg, dp_plan[c], dp_plan[c]["accountant"])
                      if account_dp_eps is not None else None)
            locals_.append(m)
        candidate = aggregate(global_model, locals_, [sizes[c] for c in clients], build_model)
        if dp is not None:
            w = max(sizes.values()) / sum(sizes.values())
            sigma = add_server_noise(candidate, dp, w)
            client_acc.add_round(sigma, w * args.clip_norm, client_dp_eps)
        if not is_finite(candidate):
            candidate.load_state_dict(prev)  # one bad round must not poison the rest
        global_model = candidate
        states.append(state_of(global_model))

    if account_dp_eps is not None:
        privacy = {str(c): {"epsilon_spent": float(d["accountant"].get_epsilon(args.delta)),
                            "sigma": d["sigma"], "sample_rate": d["sample_rate"],
                            "n_accounts": d["n_accounts"], "steps": args.rounds * d["steps_per_round"],
                            "dp_randomness": make_dp_random(args, g).name}
                   for c, d in dp_plan.items()}
    if client_acc is not None:
        privacy = client_acc.cumulative()
    metrics = select_and_test(states, clients, build_model)
    best = max(range(len(states)), key=lambda i: states[i][0])  # same first-maximum rule as select_and_test
    metrics["selected_round"] = best + (0 if include_initial else 1)
    metrics["val_pr_auc"] = states[best][0]
    return metrics, privacy


# ---------------------------------------------------------------- driver

def summarize(clients) -> dict:
    out = {"clients": {}, "pooled": {}}
    for s in ("train", "val", "test"):
        tot_n = tot_f = 0
        for c in clients:
            y = clients[c][s].example_labels
            out["clients"].setdefault(str(c), {})[s] = {"n": len(y), "fraud": int(y.sum())}
            tot_n, tot_f = tot_n + len(y), tot_f + int(y.sum())
        out["pooled"][s] = {"n": tot_n, "fraud": tot_f}
    out["train_accounts"] = {str(c): clients[c]["n_train_accounts"] for c in clients}
    return out


def mean_std(seed_results) -> tuple:
    keys = [k for k, v in next(iter(seed_results.values()))["pooled"].items() if isinstance(v, (int, float))]
    mean, std = {}, {}
    for k in keys:
        vals = [r["pooled"][k] for r in seed_results.values() if r["pooled"][k] is not None]
        mean[k] = float(np.mean(vals)) if vals else None
        std[k] = float(np.std(vals)) if vals else None
    return mean, std


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=str(ROOT / "data" / "ibm_hi_small"))
    p.add_argument("--out", default=str(ROOT / "results" / "comparison" / "ibm_hi_small.json"))
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--mu", type=float, default=0.1)
    p.add_argument("--pos-weight", type=float, default=8.0)
    p.add_argument("--oversample-ratio", type=float, default=0.02)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--dp-batch-accounts", type=int, default=256)
    p.add_argument("--dp-examples-per-account", type=int, default=4)
    p.add_argument("--max-train-accounts", type=int, default=None)
    p.add_argument("--seeded-dp", action="store_true",
                   help="seeded DP sampling and noise (tests only); default is os.urandom")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    meta = json.loads((Path(args.data_path) / "metadata" / "dataset_info.json").read_text())
    clients = {c: load_indexed_client(Path(args.data_path), c, args.max_train_accounts, seed=0)
               for c in range(meta["n_clients"])}
    if "account_dp" in args.models:
        smallest = min(clients[c]["n_train_accounts"] for c in clients)
        if args.delta >= 1.0 / smallest:
            raise SystemExit(f"delta={args.delta} must be below 1/(train accounts) = 1/{smallest} for every client")

    data = summarize(clients)
    prior = data["pooled"]["train"]["fraud"] / max(data["pooled"]["train"]["n"], 1)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset"]["num_features"] = meta["num_features"]
    config["training"]["pos_weight"] = args.pos_weight
    config["training"]["fraud_rate_prior"] = prior

    build_model = model_builder(config)

    runs = []  # (name, callable(seed, generator))
    for name in args.models:
        if name == "local":
            runs.append(("local", lambda s, g: run_local(args, clients, build_model, g)))
        elif name == "centralized":
            runs.append(("centralized", lambda s, g: run_centralized(args, clients, build_model, g)))
        elif name == "gbdt":
            runs.append(("gbdt", lambda s, g: run_gbdt(args, clients, s)))
        elif name == "fedavg":
            runs.append(("fedavg", lambda s, g: run_federated(args, clients, build_model, g, mu=0.0)))
        elif name == "fedprox":
            runs.append(("fedprox", lambda s, g: run_federated(args, clients, build_model, g, mu=args.mu)))
        elif name == "client_dp":
            for e in args.epsilons:
                runs.append((f"client_dp_eps{e}", lambda s, g, e=e: run_federated(
                    args, clients, build_model, g, mu=args.mu, client_dp_eps=e)))
        elif name == "account_dp":
            for e in args.epsilons:
                runs.append((f"account_dp_eps{e}", lambda s, g, e=e: run_federated(
                    args, clients, build_model, g, mu=args.mu, account_dp_eps=e)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"dataset": meta.get("source"), "args": vars(args), "data": data, "results": {}}
    for name, fn in runs:
        seeds = {}
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            g = torch.Generator().manual_seed(seed)
            t0 = time.time()
            metrics, privacy = fn(seed, g)
            seeds[str(seed)] = {**metrics, "privacy": privacy, "train_seconds": round(time.time() - t0, 1)}
            pm = metrics["pooled"]
            print(f"[{name} seed={seed}] pr_auc={pm['pr_auc']} recall@1%fpr={pm['recall_at_1pct_fpr']} "
                  f"f1={pm['f1']:.4f} ({seeds[str(seed)]['train_seconds']}s)", flush=True)
        mean, std = mean_std(seeds)
        results["results"][name] = {"seeds": seeds, "mean": mean, "std": std}
        out_path.write_text(json.dumps(results, indent=2))  # partial results survive a crash
    return results


if __name__ == "__main__":
    main()
