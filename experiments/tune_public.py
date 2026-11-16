"""Choose head fine-tuning settings on PUBLIC data only (Stage H), so nothing private is selected.

    python3 experiments/tune_public.py --ckpt BACKBONE.pt --data-path PUBLIC_10_CLIENTS --out tune.json
        [--epsilons 1 3 8] [--lrs 3e-4 1e-3 3e-3] [--layers last_layer classifier]
        [--rounds-grid 1 2 5 10] [any run_ibm_comparison.py training flag]

The tuning set must be a public dataset the backbone never saw, built with 10 clients. Every
(level, lr, layers, rounds) runs on its train split, with DP noise calibrated for exactly that
many rounds, and is scored by pooled validation PR-AUC of the final model. The test split is
never embedded or scored. "chosen" is what run_dp_finetune.py --settings reads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.run_dp_finetune as ft  # noqa: E402
from experiments.run_ibm_comparison import parse_args, run_federated, summarize  # noqa: E402
from src.data.sequences import load_indexed_client  # noqa: E402
from src.federated.dp_random import make_dp_random  # noqa: E402

LAYER_ORDER = {"last_layer": 0, "classifier": 1}


def choose(rows: list) -> dict:
    """Best row per level by val PR-AUC; ties go to fewer rounds, the smaller head, the smaller lr."""
    best = {}
    for r in rows:
        key = (-r["val_pr_auc"], r["rounds"], LAYER_ORDER[r["layers"]], r["lr"])
        if r["level"] not in best or key < best[r["level"]][0]:
            best[r["level"]] = (key, r)
    return {lvl: {k: r[k] for k in ("lr", "layers", "rounds", "val_pr_auc")} for lvl, (_, r) in best.items()}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--layers", nargs="+", default=["last_layer", "classifier"], choices=list(LAYER_ORDER))
    p.add_argument("--rounds-grid", type=int, nargs="+", default=[1, 2, 5, 10])
    own, rest = p.parse_known_args(argv)
    args = parse_args(rest)
    args.max_train_accounts, args.dp_fraud_rows_first = None, True  # as the HI-Small head runs
    data_path = Path(args.data_path)
    meta = json.loads((data_path / "metadata" / "dataset_info.json").read_text())
    backbone = ft.load_pretrained(Path(own.ckpt), meta["num_features"])
    clients = {c: load_indexed_client(data_path, c, None, seed=0) for c in range(meta["n_clients"])}
    smallest = min(clients[c]["n_train_accounts"] for c in clients)
    if args.delta >= 1.0 / smallest:
        raise SystemExit(f"delta={args.delta} must be below 1/(train accounts) = 1/{smallest} for every client")
    data = summarize(clients)
    prior = data["pooled"]["train"]["fraud"] / max(data["pooled"]["train"]["n"], 1)
    emb = ft.embed_clients(backbone, clients, splits=("train", "val"))
    for c in emb:
        emb[c]["test"] = emb[c]["val"]  # run_federated scores a test split; give it validation, never test
    del clients

    out_path = Path(own.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"backbone": own.ckpt, "tuned_on": args.data_path, "args": vars(args), "data": data,
              "dp_randomness": make_dp_random(args, torch.Generator()).name, "grid": [], "chosen": {}}
    levels = [None] + [float(e) for e in args.epsilons]
    for eps in levels:
        for layers in own.layers:
            build = ft.head_builder(backbone.classifier, args.pos_weight, prior, layers)
            for lr in own.lrs:
                for rounds in own.rounds_grid:
                    torch.manual_seed(0)
                    t0 = time.time()
                    a = ft.with_settings(args, {"lr": lr, "rounds": rounds})
                    metrics, _ = run_federated(a, emb, build, torch.Generator().manual_seed(0), mu=0.0,
                                               account_dp_eps=eps, include_initial=True, select="last")
                    row = {"level": "nodp" if eps is None else str(eps), "lr": lr, "layers": layers,
                           "rounds": rounds, "val_pr_auc": metrics["val_pr_auc"]}
                    result["grid"].append(row)
                    print(f"[tune {row['level']} lr={lr} {layers} R={rounds}] val_pr_auc={row['val_pr_auc']:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    result["chosen"] = choose(result["grid"])
                    out_path.write_text(json.dumps(result, indent=2))  # partial results survive a crash
    return result


if __name__ == "__main__":
    main()
