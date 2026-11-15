"""Check that a public pretraining file does not contain the private file's data.

    python3 experiments/check_public_overlap.py PUBLIC.csv PRIVATE.csv [--max-shared-row-frac 0.001]

Both files are in the raw IBM AML format. Exits 1 when more than the allowed
fraction of the public rows also appear, column for column, in the private file.
"""
import argparse
import json
from pathlib import Path


def _rows(path: Path):
    """(whole line, sender key 'bank|account') per data row, as text: bank codes keep their zero padding."""
    with open(path) as f:
        next(f)
        for line in f:
            _, bank, account, _ = line.split(",", 3)
            yield line.rstrip("\n"), bank + "|" + account


def overlap_report(public_csv: Path, private_csv: Path) -> dict:
    private_rows, private_accounts, n_private = set(), set(), 0
    for line, sender in _rows(private_csv):
        private_rows.add(hash(line))  # 64-bit hashes keep 5M rows in a few hundred MB
        private_accounts.add(sender)
        n_private += 1
    public_accounts, shared_rows, n_public = set(), 0, 0
    for line, sender in _rows(public_csv):
        shared_rows += hash(line) in private_rows
        public_accounts.add(sender)
        n_public += 1
    return {"shared_accounts": len(public_accounts & private_accounts),
            "public_accounts": len(public_accounts), "private_accounts": len(private_accounts),
            "shared_rows": shared_rows, "public_rows": n_public, "private_rows": n_private}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("public")
    p.add_argument("private")
    p.add_argument("--max-shared-row-frac", type=float, default=0.001)
    args = p.parse_args(argv)
    r = overlap_report(Path(args.public), Path(args.private))
    print(json.dumps(r, indent=2))
    frac = r["shared_rows"] / max(r["public_rows"], 1)
    if frac > args.max_shared_row_frac:
        print(f"FAIL: {frac:.4%} of the public rows are also in the private file")
        raise SystemExit(1)
    return r


if __name__ == "__main__":
    main()
