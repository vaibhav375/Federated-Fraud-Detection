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


K_HISTORY = 10
PAD_COL = 12
FEATURE_NAMES = (
    ["amount_paid_log", "amount_received_log", "same_currency", "cross_bank", "self_transfer",
     "hour_sin", "hour_cos", "dow_sin", "dow_cos", "secs_since_prev_log", "acct_tx_count_log",
     "acct_distinct_receivers_log", "is_pad"]
    + [f"fmt_{f}" for f in PAYMENT_FORMATS]
    + [f"cur_{c}" for c in CURRENCIES]
)
NUM_FEATURES = len(FEATURE_NAMES)  # 35


def _one_hot(codes: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((len(codes), width), dtype=np.float32)
    known = codes >= 0
    out[np.nonzero(known)[0], codes[known]] = 1.0
    return out


def encode_features(df: pd.DataFrame) -> np.ndarray:
    """Per-transaction features from fixed transforms only (nothing fitted on the
    data). Account features use only earlier rows of the same sending account."""
    ts = df["ts"].to_numpy()
    hour = (ts // 3600) % 24
    dow = (ts // 86400 + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    by_acct = df.groupby("from_acct", sort=False)
    prev_gap = by_acct["ts"].diff().fillna(0).to_numpy()
    tx_so_far = by_acct.cumcount().to_numpy()
    new_receiver = ~df.duplicated(["from_acct", "to_acct"])
    receivers_so_far = new_receiver.groupby(df["from_acct"], sort=False).cumsum().to_numpy() - new_receiver.to_numpy()

    base = np.column_stack([
        np.log1p(df["amount_paid"].to_numpy(np.float64)) / 20,
        np.log1p(df["amount_received"].to_numpy(np.float64)) / 20,
        df["pay_currency"].to_numpy() == df["recv_currency"].to_numpy(),
        df["from_bank"].to_numpy() != df["to_bank"].to_numpy(),
        df["from_acct"].to_numpy() == df["to_acct"].to_numpy(),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.log1p(prev_gap) / 15,
        np.log1p(tx_so_far) / 10,
        np.log1p(receivers_so_far) / 10,
        np.zeros(len(df)),
    ]).astype(np.float32)
    return np.concatenate([base,
                           _one_hot(df["pay_format"].to_numpy(), len(PAYMENT_FORMATS)),
                           _one_hot(df["pay_currency"].to_numpy(), len(CURRENCIES))], axis=1)


def _split_stats(labels: np.ndarray, accounts: np.ndarray, split: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(("train", "val", "test")):
        m = split == i
        n, fraud = int(m.sum()), int(labels[m].sum())
        out[name] = {"n": n, "fraud": fraud, "fraud_rate": fraud / n if n else 0.0,
                     "accounts": int(len(np.unique(accounts[m])))}
    return out


def build_dataset(csv_path: Path, out_dir: Path, n_clients: int = 10, k: int = K_HISTORY) -> dict:
    import json

    import torch

    from src.data.sequences import build_history_index

    out_dir = Path(out_dir)
    df = load_transactions(csv_path)
    feats = encode_features(df)
    split = time_split(df["ts"].to_numpy())
    labels = df["label"].to_numpy().astype(np.int64)
    accounts = df["from_acct"].to_numpy().astype(np.int64)
    bank_to_client = assign_banks_to_clients(df.groupby("from_bank").size(), n_clients)
    client = df["from_bank"].map(bank_to_client).to_numpy()

    meta = {"format": "indexed_v1", "source": str(csv_path), "num_features": NUM_FEATURES,
            "feature_names": FEATURE_NAMES, "k_history": k, "n_clients": n_clients,
            "n_banks": len(bank_to_client),
            "bank_to_client": {str(b): int(c) for b, c in bank_to_client.items()}, "clients": {}}
    for c in range(n_clients):
        idx = np.nonzero(client == c)[0]  # stays in time order
        hist = build_history_index(accounts[idx], k)
        cdir = out_dir / f"client_{c}"
        cdir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": torch.from_numpy(feats[idx]), "history": torch.from_numpy(hist),
                    "labels": torch.from_numpy(labels[idx]), "split": torch.from_numpy(split[idx]),
                    "account": torch.from_numpy(accounts[idx]),
                    "ts": torch.from_numpy(df["ts"].to_numpy()[idx])}, cdir / "table.pt")
        meta["clients"][str(c)] = _split_stats(labels[idx], accounts[idx], split[idx])
    meta["pooled"] = _split_stats(labels, accounts, split)
    rates = [meta["pooled"][s]["fraud_rate"] for s in ("train", "val", "test")]
    meta["fraud_rate_warning"] = bool(max(rates) > 5 * min(rates))
    if meta["fraud_rate_warning"]:
        print(f"WARNING: pooled fraud rates differ by more than 5x (train/val/test): {rates}")
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata" / "dataset_info.json").write_text(json.dumps(meta, indent=2))
    return meta
