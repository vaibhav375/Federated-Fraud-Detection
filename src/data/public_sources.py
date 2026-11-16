"""Other public AML datasets loaded into the frame load_transactions returns.

SAML-D (Oztas et al., 2023) and an AMLSim sample are separate generators from IBM AMLworld,
used as out-of-distribution public pretraining data. Fields with no IBM equivalent map to
"unknown" (an all-zero one-hot); df.attrs["unknown"] counts them per column.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ibm_aml import CURRENCIES, PAYMENT_FORMATS, _vocab_index

SAML_CURRENCY = {"UK pounds": "UK Pound", "Euro": "Euro", "Swiss franc": "Swiss Franc", "Yen": "Yen",
                 "US dollar": "US Dollar", "Mexican Peso": "Mexican Peso",
                 "Indian rupee": "Rupee", "Pakistani rupee": "Rupee"}
SAML_FORMAT = {"Credit card": "Credit Card", "Debit card": "Credit Card", "Cheque": "Cheque", "ACH": "ACH",
               "Cross-border": "Wire", "Cash Deposit": "Cash", "Cash Withdrawal": "Cash"}
AMLSIM_EPOCH = 1640995200  # 2022-01-01 00:00 UTC; AMLSim steps are read as days from here


def _frame(ts, from_bank, to_bank, from_acct, to_acct, amount, pay_cur, recv_cur, fmt, label) -> pd.DataFrame:
    df = pd.DataFrame({
        "ts": np.asarray(ts, dtype=np.int64),
        "from_bank": from_bank.astype(np.int32), "to_bank": to_bank.astype(np.int32),
        "from_acct": from_acct.astype(np.int32), "to_acct": to_acct.astype(np.int32),
        "amount_paid": np.asarray(amount, dtype=np.float32),
        "amount_received": np.asarray(amount, dtype=np.float32),
        "pay_currency": pay_cur, "recv_currency": recv_cur, "pay_format": fmt,
        "label": np.asarray(label).astype(np.int8),
    }).sort_values("ts", kind="stable").reset_index(drop=True)
    df.attrs["unknown"] = {k: int((df[k] == -1).sum()) for k in ("pay_currency", "recv_currency", "pay_format")}
    return df


def _joint_codes(a, b):
    _, codes = np.unique(np.concatenate([np.asarray(a), np.asarray(b)]), return_inverse=True)
    return codes[:len(a)], codes[len(a):]


def load_saml_d(csv_path: Path) -> pd.DataFrame:
    cat = {c: "category" for c in ("Payment_currency", "Received_currency", "Sender_bank_location",
                                   "Receiver_bank_location", "Payment_type")}
    raw = pd.read_csv(csv_path, usecols=["Time", "Date", "Sender_account", "Receiver_account", "Amount",
                                         "Payment_currency", "Received_currency", "Sender_bank_location",
                                         "Receiver_bank_location", "Payment_type", "Is_laundering"],
                      dtype={**cat, "Sender_account": np.int64, "Receiver_account": np.int64,
                             "Amount": np.float64, "Is_laundering": np.int8})
    ts = (pd.to_datetime(raw["Date"], format="%Y-%m-%d") + pd.to_timedelta(raw["Time"])).astype("int64") // 10**9
    from_bank, to_bank = _joint_codes(raw["Sender_bank_location"].astype(str), raw["Receiver_bank_location"].astype(str))
    from_acct, to_acct = _joint_codes(raw["Sender_account"], raw["Receiver_account"])
    as_ibm = lambda col, table, vocab: _vocab_index(raw[col].astype(str).map(table), vocab)  # noqa: E731
    return _frame(ts, from_bank, to_bank, from_acct, to_acct, raw["Amount"],
                  as_ibm("Payment_currency", SAML_CURRENCY, CURRENCIES),
                  as_ibm("Received_currency", SAML_CURRENCY, CURRENCIES),
                  as_ibm("Payment_type", SAML_FORMAT, PAYMENT_FORMATS), raw["Is_laundering"])


def load_amlsim(dir_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(Path(dir_path) / "transactions.csv",
                      usecols=["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TIMESTAMP", "IS_FRAUD"])
    n = len(raw)
    from_acct, to_acct = _joint_codes(raw["SENDER_ACCOUNT_ID"], raw["RECEIVER_ACCOUNT_ID"])
    label = raw["IS_FRAUD"].astype(str).str.lower().eq("true")
    const = lambda vocab, name: np.full(n, vocab.index(name), dtype=np.int8)  # noqa: E731
    zeros = np.zeros(n, dtype=np.int32)  # one bank for all
    return _frame(AMLSIM_EPOCH + raw["TIMESTAMP"].to_numpy(np.int64) * 86400, zeros, zeros, from_acct, to_acct,
                  raw["TX_AMOUNT"], const(CURRENCIES, "US Dollar"), const(CURRENCIES, "US Dollar"),
                  const(PAYMENT_FORMATS, "ACH"), label)
