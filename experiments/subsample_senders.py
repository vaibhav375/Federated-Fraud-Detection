"""Keep every row whose SENDING account falls in 1 of N hash buckets, so each kept account keeps its whole
history (sequences are keyed on the sender). Used to cut IBM LI-Medium to SAML-D size for Stage H tuning.

    python3 experiments/subsample_senders.py --zip archive.zip --member LI-Medium_Trans.csv --out out.csv [--n 3]
"""
import argparse
import csv
import io
import zipfile
import zlib


def keep(bank: str, account: str, n: int) -> bool:
    return zlib.crc32(f"{bank}|{account}".encode()) % n == 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--member", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args(argv)
    total = kept = 0
    with zipfile.ZipFile(a.zip) as z, z.open(a.member) as f, open(a.out, "w", newline="") as out:
        rows = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
        w = csv.writer(out)
        w.writerow(next(rows))
        for r in rows:
            total += 1
            if keep(r[1], r[2], a.n):
                kept += 1
                w.writerow(r)
    print(f"kept {kept} of {total} rows ({kept / total:.3f})")


if __name__ == "__main__":
    main()
