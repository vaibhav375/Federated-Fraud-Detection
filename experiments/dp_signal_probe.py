"""DP-collapse probe on one client: signal vs noise per step, and score spread after short training.

Run from the repo root:  python3 -W ignore <this file> [client]
"""
import copy, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from experiments.run_ibm_comparison import (DEFAULT_CONFIG, account_dp_plan, labels_of, model_builder,
                                            parse_args, score, summarize)
from src.data.sequences import load_indexed_client
from src.federated.dp_sgd import (AccountDPConfig, _account_groups, _per_example_grads_and_loss,
                                  _sample_rows, account_clipped_sum, train_local_account_dp)

C = int(sys.argv[1]) if len(sys.argv) > 1 else 3
args = parse_args(["--max-train-accounts", "20000", "--rounds", "10"])
meta = json.loads((Path(args.data_path) / "metadata" / "dataset_info.json").read_text())
client = load_indexed_client(Path(args.data_path), C, args.max_train_accounts, seed=0)
data = summarize({C: client})
config = copy.deepcopy(DEFAULT_CONFIG)
config["dataset"]["num_features"] = meta["num_features"]
config["training"]["pos_weight"] = args.pos_weight
config["training"]["fraud_rate_prior"] = data["pooled"]["train"]["fraud"] / data["pooled"]["train"]["n"]
build = model_builder(config)
ds, cfg = client["train"], AccountDPConfig(3.0, args.delta, args.clip_norm, args.dp_batch_accounts,
                                           args.dp_examples_per_account)
plan = account_dp_plan(args, client["n_train_accounts"], 3.0)
q, sigma = plan["sample_rate"], plan["sigma"]
d = sum(p.numel() for p in build().parameters() if p.requires_grad)

# Part A: who carries the fraud signal in one step, at the initial model
y = ds.example_labels
n_acc, inverse, order, starts, counts = _account_groups(ds)
fraud_acc = torch.zeros(n_acc, dtype=torch.bool).index_fill_(0, inverse[y == 1].unique(), True)
print(f"client {C}: {n_acc} accounts, {int(fraud_acc.sum())} with any fraud example, "
      f"{int((y == 1).sum())} fraud / {len(y)} examples; q={q:.4f} sigma={sigma:.3f} trainable d={d}")
torch.manual_seed(0)
model, g = build(), torch.Generator().manual_seed(0)
rows_tot = {"picked": 0, "fraud_acc": 0, "fraud_rows_in_sample": 0, "fraud_norm": 0.0, "legit_norm": 0.0}
S = 30
for _ in range(S):
    picked, rows, slot = _sample_rows(n_acc, inverse, order, starts, counts, q, cfg.examples_per_account, g)
    seqs, labels = ds.gather(rows)
    grads, _ = _per_example_grads_and_loss(model, seqs, labels, args.pos_weight)
    is_f = torch.zeros(len(picked), dtype=torch.bool).index_fill_(0, slot[labels == 1].unique(), True) \
        if (labels == 1).any() else torch.zeros(len(picked), dtype=torch.bool)
    for name, part in (("fraud_norm", is_f), ("legit_norm", ~is_f)):
        keep = part[slot]
        if keep.any():
            s = account_clipped_sum({k: v[keep] for k, v in grads.items()},
                                    torch.unique(slot[keep], return_inverse=True)[1], int(part.sum()), cfg.clip_norm)
            rows_tot[name] += math.sqrt(sum(float((t ** 2).sum()) for t in s.values()))
    rows_tot["picked"] += len(picked)
    rows_tot["fraud_acc"] += int(fraud_acc[picked].sum())
    rows_tot["fraud_rows_in_sample"] += int(is_f.sum())
print(f"per step (mean of {S}): accounts {rows_tot['picked']/S:.1f}, fraud accounts {rows_tot['fraud_acc']/S:.2f}, "
      f"accounts whose sampled rows include fraud {rows_tot['fraud_rows_in_sample']/S:.2f}")
print(f"  clipped-sum norm: fraud part {rows_tot['fraud_norm']/S:.3f}, legit part {rows_tot['legit_norm']/S:.3f}, "
      f"noise norm sigma*C*sqrt(d) = {sigma*args.clip_norm*math.sqrt(d):.1f}")

# Part B: score spread on validation after 2 local rounds, same code, sigma=eps3 vs sigma=0
yv = labels_of(client["val"])
for label, sig in (("sigma=0 (same pipeline, no noise)", 0.0), (f"sigma={sigma:.3f} (eps=3 plan)", sigma)):
    torch.manual_seed(0)
    m, g = build(), torch.Generator().manual_seed(0)
    t0 = time.time()
    for r in range(2):
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
        train_local_account_dp(m, ds, opt, cfg, sig, q, plan["steps_per_round"], RDPAccountant(),
                               pos_weight=args.pos_weight, generator=g)
        p = score(m, client["val"])
        print(f"{label} round {r+1}: val PR-AUC {average_precision_score(yv, p):.4f} "
              f"(base rate {yv.mean():.4f}); scores min {p.min():.4g} median {np.median(p):.4g} "
              f"max {p.max():.4g} std {p.std():.3g}; fraud mean {p[yv==1].mean():.4g} vs legit {p[yv==0].mean():.4g} "
              f"[{time.time()-t0:.0f}s]", flush=True)
