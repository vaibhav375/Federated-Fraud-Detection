"""Shared test helpers: a tiny synthetic file in the raw IBM AML format."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ibm_aml import RAW_COLUMNS, PAYMENT_FORMATS, CURRENCIES


def make_raw(n_rows=400, n_banks=6, accounts_per_bank=5, n_fraud=12, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = [(b, a) for b in range(n_banks) for a in range(accounts_per_bank)]
    sender = [pairs[i % len(pairs)] for i in range(n_rows)]
    receiver = [pairs[int(rng.integers(len(pairs)))] for _ in range(n_rows)]
    ts = pd.Timestamp("2022-09-01 00:00") + pd.to_timedelta(np.arange(n_rows), unit="min")
    amount = np.round(rng.lognormal(6, 2, n_rows), 2)
    label = np.zeros(n_rows, dtype=int)
    if n_fraud:
        label[[i * (n_rows // n_fraud) for i in range(n_fraud)]] = 1
    return pd.DataFrame({
        "Timestamp": ts.strftime("%Y/%m/%d %H:%M"),
        "From Bank": [f"{b + 1:03d}" for b, _ in sender],
        "Account": [f"A{b}_{a}" for b, a in sender],
        "To Bank": [f"{b + 1:03d}" for b, _ in receiver],
        "Account.1": [f"A{b}_{a}" for b, a in receiver],
        "Amount Received": amount,
        "Receiving Currency": [CURRENCIES[i % 3] for i in range(n_rows)],
        "Amount Paid": amount,
        "Payment Currency": [CURRENCIES[i % 2] for i in range(n_rows)],
        "Payment Format": [PAYMENT_FORMATS[i % len(PAYMENT_FORMATS)] for i in range(n_rows)],
        "Is Laundering": label,
    })[RAW_COLUMNS]


def write_raw(df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write with the duplicate 'Account' header the real file has."""
    path = Path(tmp_path) / "raw.csv"
    header = ",".join(c if c != "Account.1" else "Account" for c in df.columns)
    with open(path, "w") as f:
        f.write(header + "\n")
        df.to_csv(f, header=False, index=False)
    return path
