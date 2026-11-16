"""Build the indexed IBM AML dataset: one time-sorted table per bank group.

    python3 experiments/build_ibm_aml.py [--src RAW.csv] [--out data/ibm_hi_small] [--clients 10]
        [--source ibm|saml_d|amlsim] [--group-by bank|account_hash]

The raw file path defaults to $IBM_AML_RAW, else the project drive's copy.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ibm_aml import build_dataset  # noqa: E402

DEFAULT_RAW = os.environ.get("IBM_AML_RAW",
                             "/Volumes/Untitled/datasets_capstone/ibm_aml/hi_small/HI-Small_Trans.csv")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=DEFAULT_RAW)
    p.add_argument("--out", default=str(ROOT / "data" / "ibm_hi_small"))
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--source", default="ibm", choices=["ibm", "saml_d", "amlsim"])
    p.add_argument("--group-by", default="bank", choices=["bank", "account_hash"])
    args = p.parse_args(argv)
    meta = build_dataset(Path(args.src), Path(args.out), n_clients=args.clients,
                         source=args.source, group_by=args.group_by)
    print(f"{meta['n_banks']} {'banks' if args.group_by == 'bank' else 'accounts'} -> {args.clients} clients; "
          f"unknown categories {meta['unknown_categories']}")
    print(f"{'client':>6} {'split':>5} {'n':>10} {'fraud':>7} {'rate':>8} {'accounts':>9}")
    for c, splits in list(meta["clients"].items()) + [("all", meta["pooled"])]:
        for s, v in splits.items():
            print(f"{c:>6} {s:>5} {v['n']:>10} {v['fraud']:>7} {v['fraud_rate']:>8.5f} {v['accounts']:>9}")
    return meta


if __name__ == "__main__":
    main()
