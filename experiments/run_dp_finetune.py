"""Federated fine-tuning of a publicly pretrained model on the private IBM set.

    python3 experiments/run_dp_finetune.py --ckpt checkpoints/pretrain_li_small.pt
        [--models zero_shot head_nodp full_nodp head_dp] [--dp-train classifier|last_layer]
        [--mu 0] [--out results/comparison/ibm_hi_small_a4.json]
        [--settings tuned.json] [--save-scores DIR]
        [any flag of run_ibm_comparison.py: --epsilons --seeds --rounds --lr --delta ...]

--settings  head runs take lr, rounds and trained layers from a tune_public.py file (per epsilon,
            and "nodp" for head_nodp) and report the model after exactly those rounds: nothing is
            selected on the private validation split. Without it, A4's validation selection applies.

zero_shot  the pretrained model as is; it never trains on the private data
head_nodp  FedAvg on the classifier head only, no DP
full_nodp  FedAvg on the whole pretrained model, no DP, at 20,000 train accounts per
           client like Stage A0: the no-DP ceiling with the same public data
head_dp    account-level DP-SGD on the classifier head (or only its last layer)

The frozen backbone turns each example into its 32-wide embedding once, in eval mode,
so an example's embedding depends only on its own account's rows; head training then
runs on those vectors. DP runs use every train account and take an account's fraud
rows first. The starting model is selectable as round 0 on validation.
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
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_ibm_comparison import (SCORE_BATCH, evaluate_split, final_metrics, mean_std,  # noqa: E402
                                            parse_args, run_federated, summarize, val_pr_auc)
from src.data.sequences import load_indexed_client  # noqa: E402
from src.federated.dp_random import make_dp_random  # noqa: E402
from src.federated.dp_sgd import freeze_non_dp_heads  # noqa: E402
from src.models.temporal_graph_transformer import TemporalGraphTransformer  # noqa: E402

MODELS = ["zero_shot", "head_nodp", "full_nodp", "head_dp"]
FULL_NODP_ACCOUNTS = 20000  # Stage A0's cap, so full_nodp is comparable to A0's fedavg


def load_pretrained(path: Path, num_features: int) -> TemporalGraphTransformer:
    saved = torch.load(path, weights_only=False)
    if saved["num_features"] != num_features:
        raise ValueError(f"checkpoint {path} was trained on {saved['num_features']} features, "
                         f"the data has {num_features}")
    model = TemporalGraphTransformer(saved["config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model


def embed(backbone, ds) -> torch.Tensor:
    backbone.eval()  # dropout off: each embedding depends only on its own example
    out = []
    with torch.no_grad():
        for s in range(0, len(ds), SCORE_BATCH):
            seqs, _ = ds.gather(torch.arange(s, min(s + SCORE_BATCH, len(ds))))
            out.append(backbone(seqs, labels=None)["embeddings"])
    return torch.cat(out) if out else torch.zeros(0, 32)


class EmbeddingDataset:
    """Same contract as SequenceIndexDataset, over precomputed embeddings."""

    def __init__(self, emb, labels, accounts):
        self.emb, self.example_labels, self.example_accounts = emb, labels, accounts

    def __len__(self):
        return len(self.emb)

    def gather(self, idx):
        return self.emb[idx], self.example_labels[idx]


def set_prior_bias(model, prior: float):
    """Final bias = log-odds of the (public) pooled train fraud rate, as at model init."""
    prior = min(max(prior, 1e-4), 1 - 1e-4)
    with torch.no_grad():
        model.classifier[3].bias.fill_(math.log(prior / (1 - prior)))


class HeadModel(nn.Module):
    """The classifier alone, with the interface the training and scoring code expects."""

    def __init__(self, classifier, pos_weight: float):
        super().__init__()
        self.classifier = copy.deepcopy(classifier)
        self.pos_weight = pos_weight

    def forward(self, x, graph_features=None, labels=None):
        return {"fraud_prob": torch.clamp(self.classifier(x).squeeze(-1), 1e-7, 1 - 1e-7)}

    def compute_loss(self, outputs, labels):
        w = torch.where(labels == 1, self.pos_weight, 1.0)
        loss = F.binary_cross_entropy(outputs["fraud_prob"], labels.float(), weight=w)
        return loss, {"bce": loss.item()}

    def reset_bias(self, prior: float):
        set_prior_bias(self, prior)

    def set_trainable(self, which: str):
        for p in self.parameters():
            p.requires_grad_(which == "classifier")
        if which == "last_layer":
            for p in self.classifier[3].parameters():
                p.requires_grad_(True)
        elif which != "classifier":
            raise ValueError(f"unknown trainable set {which!r}")


def embed_clients(backbone, clients) -> dict:
    """{client: {split: EmbeddingDataset}, plus n_train_accounts} from the frozen backbone."""
    emb = {c: {s: EmbeddingDataset(embed(backbone, clients[c][s]), clients[c][s].example_labels,
                                   clients[c][s].example_accounts) for s in ("train", "val", "test")}
           for c in clients}
    for c in clients:
        emb[c]["n_train_accounts"] = clients[c]["n_train_accounts"]
    return emb


def load_settings(path, models, epsilons) -> dict:
    """The tuned file's {"chosen": ...}; exits before any training if a needed level is missing."""
    settings = json.loads(Path(path).read_text())
    needed = (["nodp"] if "head_nodp" in models else []) + \
             ([str(float(e)) for e in epsilons] if "head_dp" in models else [])
    for key in needed:
        if key not in settings.get("chosen", {}):
            raise SystemExit(f"tuned settings have no entry for {key}")
    return settings


def with_settings(args, s):
    a = copy.copy(args)
    a.lr, a.rounds = s["lr"], s["rounds"]  # DP noise is calibrated to a.rounds
    return a


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--dp-train", default="classifier", choices=["classifier", "last_layer"])
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--out", default=str(ROOT / "results" / "comparison" / "ibm_hi_small_a4.json"))
    p.add_argument("--settings", default=None)
    p.add_argument("--save-scores", default=None)
    own, rest = p.parse_known_args(argv)
    args = parse_args(rest)
    settings = load_settings(own.settings, own.models, args.epsilons) if own.settings else None
    select = "last" if settings else "val"
    args.mu, args.out, args.max_train_accounts = own.mu, own.out, None  # DP and head runs use every account
    args.dp_fraud_rows_first = True
    data_path = Path(args.data_path)
    meta = json.loads((data_path / "metadata" / "dataset_info.json").read_text())
    backbone = load_pretrained(Path(own.ckpt), meta["num_features"])
    n_clients = range(meta["n_clients"])

    clients = {c: load_indexed_client(data_path, c, None, seed=0) for c in n_clients}
    if "head_dp" in own.models:
        smallest = min(clients[c]["n_train_accounts"] for c in clients)
        if args.delta >= 1.0 / smallest:
            raise SystemExit(f"delta={args.delta} must be below 1/(train accounts) = 1/{smallest} for every client")
    data = summarize(clients)
    prior = data["pooled"]["train"]["fraud"] / max(data["pooled"]["train"]["n"], 1)

    def head_builder(which):
        def build():
            h = HeadModel(backbone.classifier, args.pos_weight)
            h.reset_bias(prior)
            h.set_trainable(which)
            return h
        return build

    def full_build():
        m = copy.deepcopy(backbone)
        freeze_non_dp_heads(m)
        set_prior_bias(m, prior)  # same public prior as the head runs
        return m

    out_path = Path(own.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"dataset": meta.get("source"), "args": vars(args), "data": data, "results": {},
               "pretrained": {"ckpt": own.ckpt, "num_features": meta["num_features"]},
               "dp_train": own.dp_train, "trained_parameters": {},
               "selection": "fixed_rounds" if settings else "val", "settings": settings,
               "dp_randomness": make_dp_random(args, torch.Generator()).name}

    def record(name, fn, seeds):
        per_seed = {}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            metrics, privacy = fn(torch.Generator().manual_seed(seed))
            probs = metrics.pop("test_probs", None)
            if own.save_scores and probs is not None:
                Path(own.save_scores).mkdir(parents=True, exist_ok=True)
                np.savez(Path(own.save_scores) / f"{name}_seed{seed}.npz",
                         **{f"client_{c}": np.asarray(v) for c, v in probs.items()})
            per_seed[str(seed)] = {**metrics, "privacy": privacy, "train_seconds": round(time.time() - t0, 1)}
            pm = metrics["pooled"]
            print(f"[{name} seed={seed}] pr_auc={pm['pr_auc']} recall@1%fpr={pm['recall_at_1pct_fpr']} "
                  f"val_pr_auc={metrics['val_pr_auc']:.4f} selected_round={metrics['selected_round']} "
                  f"({per_seed[str(seed)]['train_seconds']}s)", flush=True)
        mean, std = mean_std(per_seed)
        results["results"][name] = {"seeds": per_seed, "mean": mean, "std": std}
        out_path.write_text(json.dumps(results, indent=2))  # partial results survive a crash

    if "full_nodp" in own.models:  # the only capped run; its tables are dropped before embedding
        capped = {c: load_indexed_client(data_path, c, FULL_NODP_ACCOUNTS, seed=0) for c in n_clients}
        results["trained_parameters"]["full_nodp"] = n_trainable(full_build())
        record("full_nodp", lambda g: run_federated(args, capped, full_build, g, mu=own.mu, include_initial=True),
               args.seeds)
        del capped

    if "zero_shot" in own.models:
        def zero_shot(g):
            models = {c: backbone for c in clients}
            val = evaluate_split(models, clients, "val")
            m = final_metrics(val, evaluate_split(models, clients, "test"), clients)
            m["selected_round"], m["val_pr_auc"] = 0, val_pr_auc(val, clients)
            return m, None
        record("zero_shot", zero_shot, args.seeds[:1])  # deterministic, one entry is enough

    if {"head_nodp", "head_dp"} & set(own.models):
        t0 = time.time()
        emb = embed_clients(backbone, clients)
        del clients
        print(f"embedded {sum(len(emb[c][s]) for c in emb for s in ('train', 'val', 'test'))} examples "
              f"in {time.time() - t0:.0f}s", flush=True)
        keep = bool(own.save_scores)
        if "head_nodp" in own.models:
            s = settings["chosen"]["nodp"] if settings else None
            a, build = (with_settings(args, s), head_builder(s["layers"])) if s else (args, head_builder("classifier"))
            results["trained_parameters"]["head_nodp"] = n_trainable(build())
            record("head_nodp", lambda g: run_federated(a, emb, build, g, mu=own.mu, include_initial=True,
                                                        select=select, return_probs=keep), args.seeds)
        if "head_dp" in own.models:
            for e in args.epsilons:
                s = settings["chosen"][str(float(e))] if settings else None
                a, build = (with_settings(args, s), head_builder(s["layers"])) if s else (args, head_builder(own.dp_train))
                key = f"head_dp_eps{e}" if settings else "head_dp"
                results["trained_parameters"][key] = n_trainable(build())
                print(f"{key} trains {results['trained_parameters'][key]} parameters", flush=True)
                record(f"head_dp_eps{e}", lambda g, a=a, build=build, e=e: run_federated(
                    a, emb, build, g, mu=own.mu, account_dp_eps=e, include_initial=True,
                    select=select, return_probs=keep), args.seeds)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
