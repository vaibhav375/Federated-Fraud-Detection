"""IBM AML (AMLworld) transactions: loading, bank grouping, time split, features.

Raw format: one row per transfer, sender bank/account and receiver bank/account,
amounts in both currencies, payment format and a laundering label. A transaction
is owned by its sending bank.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
               "Amount Received", "Receiving Currency", "Amount Paid",
               "Payment Currency", "Payment Format", "Is Laundering"]
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["US Dollar", "Euro", "Yuan", "Yen", "UK Pound", "Rupee", "Ruble",
              "Canadian Dollar", "Australian Dollar", "Swiss Franc", "Brazil Real",
              "Mexican Peso", "Saudi Riyal", "Shekel", "Bitcoin"]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def _vocab_index(values: pd.Series, vocab: list) -> np.ndarray:
    # Unknown values map to -1 so a new category never breaks the build.
    return pd.Categorical(values, categories=vocab).codes.astype(np.int8)


def load_transactions(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path,
        dtype={"From Bank": str, "Account": str, "To Bank": str, "Account.1": str,
               "Amount Received": np.float64, "Amount Paid": np.float64,
               "Receiving Currency": str, "Payment Currency": str,
               "Payment Format": str, "Is Laundering": np.int8},
    )
    raw.columns = RAW_COLUMNS  # pandas renames the duplicate header to Account.1 already
    ts = pd.to_datetime(raw["Timestamp"], format=TIMESTAMP_FORMAT).astype("int64") // 10**9

    # Bank codes are zero-padded strings ("010" != "10"), so factorize the strings.
    banks, bank_codes = np.unique(np.concatenate([raw["From Bank"].values, raw["To Bank"].values]),
                                  return_inverse=True)
    n = len(raw)
    # An account is the pair (bank, account string); one code space for both sides.
    acct_keys = np.concatenate([raw["From Bank"].values + "|" + raw["Account"].values,
                                raw["To Bank"].values + "|" + raw["Account.1"].values])
    _, acct_codes = np.unique(acct_keys, return_inverse=True)

    df = pd.DataFrame({
        "ts": ts.values.astype(np.int64),
        "from_bank": bank_codes[:n].astype(np.int32),
        "to_bank": bank_codes[n:].astype(np.int32),
        "from_acct": acct_codes[:n].astype(np.int32),
        "to_acct": acct_codes[n:].astype(np.int32),
        "amount_paid": raw["Amount Paid"].values.astype(np.float32),
        "amount_received": raw["Amount Received"].values.astype(np.float32),
        "pay_currency": _vocab_index(raw["Payment Currency"], CURRENCIES),
        "recv_currency": _vocab_index(raw["Receiving Currency"], CURRENCIES),
        "pay_format": _vocab_index(raw["Payment Format"], PAYMENT_FORMATS),
        "label": raw["Is Laundering"].values.astype(np.int8),
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def assign_banks_to_clients(bank_tx_counts: pd.Series, n_clients: int = 10) -> dict:
    """Greedy balancing: largest bank first, each to the currently lightest client."""
    order = sorted(bank_tx_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    totals = [0] * n_clients
    out = {}
    for bank, count in order:
        client = min(range(n_clients), key=lambda c: (totals[c], c))
        out[bank] = client
        totals[client] += count
    return out


def time_split(ts: np.ndarray, train_frac: float = 0.6, val_frac: float = 0.2) -> np.ndarray:
    """0 = train, 1 = val, 2 = test, by global time cut-offs (ties stay together)."""
    ts = np.asarray(ts)
    n = len(ts)
    t1 = ts[int(n * train_frac)]
    t2 = ts[min(int(n * (train_frac + val_frac)), n - 1)]
    return np.where(ts < t1, 0, np.where(ts < t2, 1, 2)).astype(np.int8)
