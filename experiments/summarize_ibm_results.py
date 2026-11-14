"""Print the Stage A0 results table (mean ± std over seeds) from the runner's JSON.

    python3 experiments/summarize_ibm_results.py results/comparison/ibm_hi_small.json
"""
import json
import sys


def fmt(mean, std, key):
    m, s = mean.get(key), std.get(key)
    if m is None:
        return "n/a"
    return f"{m:.3f} ± {s:.3f}" if s else f"{m:.3f}"


def main(path):
    r = json.load(open(path))
    print("| Model | PR-AUC | Recall @ 1% FPR | F1 | Train time (s) | ε spent |")
    print("|---|---|---|---|---|---|")
    for name, res in r["results"].items():
        seed0 = next(iter(res["seeds"].values()))
        priv = seed0["privacy"]
        if priv is None:
            eps = "none"
        elif "epsilon_rdp_total" in priv:
            eps = f"{priv['epsilon_rdp_total']:.1f} (client-level, composed)"
        else:
            eps = f"max {max(v['epsilon_spent'] for v in priv.values()):.2f} (account-level)"
        t = sum(s["train_seconds"] for s in res["seeds"].values()) / len(res["seeds"])
        print(f"| {name} | {fmt(res['mean'], res['std'], 'pr_auc')} | "
              f"{fmt(res['mean'], res['std'], 'recall_at_1pct_fpr')} | {fmt(res['mean'], res['std'], 'f1')} | "
              f"{t:.0f} | {eps} |")
    if "local" in r["results"] and "fedprox" in r["results"]:
        loc = r["results"]["local"]["seeds"]
        fed = r["results"]["fedprox"]["seeds"]
        wins = []
        for seed in fed:
            lc, fc = loc[seed]["per_client"], fed[seed]["per_client"]
            wins.append(sum((fc[c]["pr_auc"] or 0) > (lc[c]["pr_auc"] or 0) for c in fc))
        print(f"\nfedprox beats local on per-client test PR-AUC for {wins} of {len(fc)} clients (per seed)")


if __name__ == "__main__":
    main(sys.argv[1])
