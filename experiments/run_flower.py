"""Stage B1: A0's FedProx (optionally with account-level DP) as a Flower app with SecAgg+.

    python3 experiments/run_flower.py [--rounds 10] [--account-dp-eps 8] [--no-secagg]
        [--debug-dump-locals] [any run_ibm_comparison.py training flag]
        [--ckpt PRETRAINED.pt [--head-train last_layer|classifier] [--settings tuned.json]]

--ckpt switches to head-DP mode (Stage H): banks embed their data with the frozen public backbone
and train only the classifier head, every train account, fraud rows first (as A4). With
--settings, lr, rounds and head layers come from the tuned file (the --account-dp-eps level, or
"nodp") and the model after the last round is reported.

With --debug-dump-locals each round's aggregate is checked against the plain
weighted average of the banks' local models (debug only: the dumps defeat SecAgg+).
Model selection and test metrics use the A0 code, so rows compare with A0's JSON.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from opacus.accountants import RDPAccountant

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_ibm_comparison import (account_dp_plan, evaluate_split, evaluate_state,  # noqa: E402
                                            select_and_test, val_pr_auc)
from src.federated.dp_random import make_dp_random  # noqa: E402
from src.federated.flower_app import (FlowerSettings, aggregation_errors, build_model_for,  # noqa: E402
                                      prepare_head_embeddings,
                                      client_data, client_sizes, initial_model, run,
                                      secagg_error_bound, set_trainable)


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=str(ROOT / "data" / "ibm_hi_small"))
    p.add_argument("--out", default=str(ROOT / "results" / "flower" / "b1.json"))
    p.add_argument("--work-dir", default=str(ROOT / "results" / "flower" / "run"))
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--account-dp-eps", type=float, default=None)
    p.add_argument("--no-secagg", action="store_true")
    p.add_argument("--debug-dump-locals", action="store_true")
    p.add_argument("--clipping-range", type=float, default=16.0)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--head-train", default="last_layer", choices=["last_layer", "classifier"])
    p.add_argument("--settings", default=None)
    args, rest = p.parse_known_args(argv)
    args.tuned = None
    if args.settings:
        key = "nodp" if args.account_dp_eps is None else str(float(args.account_dp_eps))
        chosen = json.loads(Path(args.settings).read_text()).get("chosen", {})
        if key not in chosen:
            raise SystemExit(f"tuned settings have no entry for {key}")
        args.tuned = chosen[key]
        args.rounds, args.head_train = args.tuned["rounds"], args.tuned["layers"]
        rest += ["--lr", str(args.tuned["lr"])]
    if args.ckpt is None and not any(x == "--max-train-accounts" or x.startswith("--max-train-accounts=")
                                     for x in rest):
        rest += ["--max-train-accounts", "20000"]
    return args, rest + ["--rounds", str(args.rounds)]  # DP noise is calibrated to the round count


def privacy_report(settings, clients) -> dict:
    """Same per-bank fields as A0, from public plan values (sigma, q, steps)."""
    out = {}
    for c, data in clients.items():
        d = account_dp_plan(settings.args, data["n_train_accounts"], settings.account_dp_eps)
        steps = settings.args.rounds * d["steps_per_round"]
        acc = RDPAccountant()
        acc.history = [(d["sigma"], d["sample_rate"], steps)]
        out[str(c)] = {"epsilon_spent": float(acc.get_epsilon(settings.args.delta)), "sigma": d["sigma"],
                       "sample_rate": d["sample_rate"], "n_accounts": d["n_accounts"], "steps": steps}
    return out


def main(argv=None) -> dict:
    a, run_args = parse(argv)
    meta = json.loads((Path(a.data_path) / "metadata" / "dataset_info.json").read_text())
    s = FlowerSettings(data_path=a.data_path, run_args=run_args, n_clients=meta["n_clients"], seed=a.seed,
                       account_dp_eps=a.account_dp_eps, secagg=not a.no_secagg, clipping_range=a.clipping_range,
                       out_dir=a.work_dir, dump_locals=a.debug_dump_locals, ckpt=a.ckpt, head_train=a.head_train)
    if s.ckpt:
        prepare_head_embeddings(s)  # always fresh: a work dir may hold another backbone's embeddings
    if s.account_dp_eps is not None:
        smallest = min(client_data(s, c)["n_train_accounts"] for c in range(s.n_clients))
        if s.args.delta >= 1.0 / smallest:
            raise SystemExit(f"delta={s.args.delta} must be below 1/(train accounts) = 1/{smallest} for every client")
    start = time.time()
    run(s, a.rounds)
    glob = Path(a.work_dir) / "global"
    stamps = [start] + [(glob / f"round{r}.npz").stat().st_mtime for r in range(1, a.rounds + 1)]

    clients = {c: client_data(s, c) for c in range(s.n_clients)}
    build = build_model_for(s)
    states = []
    for r in range(1, a.rounds + 1):
        m = initial_model(build, s.seed)
        arrays = np.load(glob / f"round{r}.npz")
        set_trainable(m, [arrays[k] for k in arrays.files])
        states.append((val_pr_auc(evaluate_split({c: m for c in clients}, clients, "val"), clients),
                       {c: m.state_dict() for c in clients}))

    result = {
        "settings": dataclasses.asdict(s), "secagg": s.secagg, "debug_dump_locals": s.dump_locals,
        "rounds": a.rounds, "round_seconds": [round(t1 - t0, 2) for t0, t1 in zip(stamps, stamps[1:])],
        "aggregation_errors": aggregation_errors(a.work_dir, client_sizes(s), a.rounds) if s.dump_locals else None,
        "error_bound": secagg_error_bound(s.n_clients, s.clipping_range) if s.secagg else 1e-5,
        "metrics": (evaluate_state(states[-1][1], clients, build) if a.tuned
                    else select_and_test(states, clients, build)),
        "selection": "fixed_rounds" if a.tuned else "val",
        "mode": "head_dp" if s.ckpt else "full", "ckpt": s.ckpt, "head_train": s.head_train if s.ckpt else None,
        "tuned": a.tuned, "dp_randomness": make_dp_random(s.args, torch.Generator()).name,
        "privacy": privacy_report(s, clients) if s.account_dp_eps is not None else None,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    pm = result["metrics"]["pooled"]
    print(f"[flower secagg={s.secagg} dp_eps={s.account_dp_eps}] pr_auc={pm['pr_auc']} "
          f"recall@1%fpr={pm['recall_at_1pct_fpr']} max_agg_error={max(result['aggregation_errors'] or [0])}",
          flush=True)
    return result


if __name__ == "__main__":
    main()
