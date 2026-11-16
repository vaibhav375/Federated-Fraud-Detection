"""Per-bank breakdown of one client's test scores (Stage H: why client 0 scores ~0.02).

    python3 experiments/analyze_client0.py --raw HI-Small_Trans.csv --scores SCORES.npz
        [--client 0] [--n-clients 10] [--out results/hardening/client0.json]

Rebuilds the client assignment and time split exactly as build_dataset does, so the client's
rows line up with the saved test scores (run_dp_finetune.py --save-scores, test-split order).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ibm_aml import assign_banks_to_clients, load_transactions, time_split  # noqa: E402


def per_bank_report(df_client, split: np.ndarray, test_probs: np.ndarray) -> list:
    """One row per sending bank of this client; test_probs follow the client's test rows in order."""
    banks = df_client["from_bank"].to_numpy()
    labels = df_client["label"].to_numpy().astype(int)
    test = split == 2
    probs = np.full(len(banks), np.nan)
    probs[test] = test_probs
    rows = []
    for b in np.unique(banks):
        m = banks == b
        tr, te = m & (split == 0), m & test
        y = labels[te]
        rows.append({"bank": int(b), "train_n": int(tr.sum()), "train_fraud": int(labels[tr].sum()),
                     "test_n": int(te.sum()), "test_fraud": int(y.sum()),
                     "share_of_client": float(m.mean()),
                     "test_pr_auc": float(average_precision_score(y, probs[te])) if 0 < y.sum() < len(y) else None})
    return sorted(rows, key=lambda r: (-r["test_n"], r["bank"]))


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--client", type=int, default=0)
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--out", default=str(ROOT / "results" / "hardening" / "client0.json"))
    a = p.parse_args(argv)
    df = load_transactions(Path(a.raw))
    client = df["from_bank"].map(assign_banks_to_clients(df.groupby("from_bank").size(), a.n_clients)).to_numpy()
    split = time_split(df["ts"].to_numpy())
    idx = np.nonzero(client == a.client)[0]  # time order, as build_dataset stores it
    probs = np.load(a.scores)[f"client_{a.client}"]
    n_test = int((split[idx] == 2).sum())
    if len(probs) != n_test:
        raise ValueError(f"scores have {len(probs)} rows but client {a.client} has {n_test} test rows")
    df_c, split_c = df.iloc[idx].reset_index(drop=True), split[idx]
    banks = per_bank_report(df_c, split_c, probs)
    y = df_c["label"].to_numpy()
    out = {"client": a.client, "n_rows": len(idx), "n_banks": len(banks),
           "largest_bank_share": max(r["share_of_client"] for r in banks),
           "fraud_rate": {s: float(y[split_c == i].mean()) if (split_c == i).any() else None
                          for i, s in enumerate(("train", "val", "test"))},
           "banks": banks, "scores": a.scores}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
