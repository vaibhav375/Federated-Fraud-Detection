"""Pretrain the deep model, without DP, on a public dataset built by build_ibm_aml.py (one client).

    python3 experiments/pretrain_public.py [--data-path data/ibm_li_small] [--epochs 5] [--seed 0]
        [--out-ckpt checkpoints/pretrain_li_small.pt] [--out-json results/pretrain/li_small.json]
        [any training flag of run_ibm_comparison.py, e.g. --max-train-accounts N --lr 1e-3]

The public set is outside the privacy guarantee: nothing here touches the private data.
The best epoch is chosen on the public validation split; its test split is scored once.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_ibm_comparison import (DEFAULT_CONFIG, BatchIter, is_finite, labels_of,  # noqa: E402
                                            model_builder, parse_args, score, snapshot, trainable)
from run_training import train_local  # noqa: E402
from src.data.sequences import load_indexed_client  # noqa: E402
from src.utils.metrics import best_f1_threshold, fraud_metrics  # noqa: E402


def pretrain(data_path: Path, epochs: int, seed: int, args) -> dict:
    client = load_indexed_client(Path(data_path), 0, args.max_train_accounts, seed=0)
    y = client["train"].example_labels
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset"]["num_features"] = client["num_features"]
    config["training"]["pos_weight"] = args.pos_weight
    config["training"]["fraud_rate_prior"] = float(y.float().mean())

    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = model_builder(config)()
    opt = torch.optim.Adam(trainable(m), lr=args.lr, weight_decay=1e-5)
    yv = labels_of(client["val"])
    vals, best_score, best_state, best_epoch, t0 = [], -2.0, None, 0, time.time()
    for e in range(1, epochs + 1):
        train_local(m, BatchIter(client["train"], args.batch_size, args.oversample_ratio, g),
                    opt, None, None, "cpu", global_params=None, mu=0.0, epochs=1)
        s = fraud_metrics(yv, score(m, client["val"]))["pr_auc"] or -1.0
        vals.append(s)
        print(f"[pretrain epoch {e}/{epochs}] val pr_auc={s:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if s > best_score and is_finite(m):
            best_score, best_state, best_epoch = s, snapshot(m), e
    if best_state is None:
        raise RuntimeError("pretraining produced no finite model")
    m.load_state_dict(best_state)
    thr = best_f1_threshold(yv, score(m, client["val"]))
    return {"state_dict": best_state, "config": config, "num_features": client["num_features"],
            "val_pr_auc_by_epoch": vals, "best_epoch": best_epoch,
            "test": fraud_metrics(labels_of(client["test"]), score(m, client["test"]), thr),
            "train_seconds": round(time.time() - t0, 1),
            "data": {s: {"n": len(client[s]), "fraud": int(client[s].example_labels.sum())}
                     for s in ("train", "val", "test")},
            "train_accounts": client["n_train_accounts"]}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=str(ROOT / "data" / "ibm_li_small"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-ckpt", default=str(ROOT / "checkpoints" / "pretrain_li_small.pt"))
    p.add_argument("--out-json", default=str(ROOT / "results" / "pretrain" / "li_small.json"))
    own, rest = p.parse_known_args(argv)
    args = parse_args(["--data-path", own.data_path, *rest])
    out = pretrain(Path(own.data_path), own.epochs, own.seed, args)
    for path in (own.out_ckpt, own.out_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["state_dict"], "config": out["config"],
                "num_features": out["num_features"], "source": own.data_path}, own.out_ckpt)
    report = {k: v for k, v in out.items() if k != "state_dict"}
    report.update(source=own.data_path, epochs=own.epochs, seed=own.seed, args=vars(args))
    Path(own.out_json).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
