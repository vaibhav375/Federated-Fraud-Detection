import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.ibm_aml import CURRENCIES, PAYMENT_FORMATS, build_dataset
from src.data.public_sources import load_amlsim, load_saml_d

SAML_COLS = ["Time", "Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
             "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type",
             "Is_laundering", "Laundering_type"]
REAL_SAML_D = Path("/Volumes/Untitled/datasets_capstone/SAML-D.csv")


def _write_saml(rows, path):
    pd.DataFrame(rows, columns=SAML_COLS).to_csv(path, index=False)
    return path


def make_saml(n_rows=60, n_accounts=12, n_fraud=6, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-10-07 08:00:00")
    rows = []
    for i in range(n_rows):
        ts = t0 + pd.Timedelta(minutes=37 * i)
        rows.append([ts.strftime("%H:%M:%S"), ts.strftime("%Y-%m-%d"), 1000 + i % n_accounts,
                     1000 + int(rng.integers(n_accounts)), float(np.round(rng.lognormal(7, 1), 2)),
                     "UK pounds", "UK pounds", "UK", "UK", "Credit card",
                     int(i % (n_rows // n_fraud) == 0), "x"])
    return rows


def test_saml_d_mapping(tmp_path):
    rows = [
        ["10:35:19", "2022-10-07", 11, 22, 100.0, "UK pounds", "UK pounds", "UK", "UK", "Cash Deposit", 0, "n"],
        ["10:35:20", "2022-10-07", 11, 33, 250.5, "UK pounds", "Dirham", "UK", "UAE", "Cross-border", 1, "l"],
        ["10:36:00", "2022-10-07", 22, 11, 75.0, "Euro", "Euro", "UK", "UK", "Debit card", 0, "n"],
        ["10:37:00", "2022-10-07", 33, 22, 10.0, "Pakistani rupee", "Indian rupee", "Pakistan", "UK", "ACH", 0, "n"],
        ["10:38:00", "2022-10-07", 22, 33, 12.0, "US dollar", "Swiss franc", "UK", "UK", "Cheque", 0, "n"],
        ["10:39:00", "2022-10-07", 44, 11, 13.0, "Yen", "Mexican Peso", "UK", "UK", "Cash Withdrawal", 0, "n"],
    ]
    df = load_saml_d(_write_saml(rows, tmp_path / "s.csv"))
    assert len(df) == 6 and int(df["ts"].iloc[0]) == 1665138919
    assert list(df["label"]) == [0, 1, 0, 0, 0, 0]
    assert (df["amount_paid"] == df["amount_received"]).all()
    assert df["amount_paid"].iloc[1] == pytest.approx(250.5)
    uk = CURRENCIES.index("UK Pound")
    assert df["pay_currency"].iloc[0] == uk and df["recv_currency"].iloc[1] == -1  # Dirham is unknown
    assert df.attrs["unknown"] == {"pay_currency": 0, "recv_currency": 1, "pay_format": 0}
    assert df["pay_currency"].iloc[3] == CURRENCIES.index("Rupee") == df["recv_currency"].iloc[3]
    assert df["recv_currency"].iloc[5] == CURRENCIES.index("Mexican Peso")
    fmt = lambda name: PAYMENT_FORMATS.index(name)  # noqa: E731
    assert list(df["pay_format"]) == [fmt("Cash"), fmt("Wire"), fmt("Credit Card"), fmt("ACH"),
                                      fmt("Cheque"), fmt("Cash")]
    assert df["from_bank"].iloc[1] != df["to_bank"].iloc[1]  # UK -> UAE crosses banks
    assert df["from_bank"].iloc[0] == df["to_bank"].iloc[0]
    assert df["from_acct"].iloc[0] == df["from_acct"].iloc[1] == df["to_acct"].iloc[2]  # account 11


@pytest.mark.skipif(not REAL_SAML_D.exists(), reason="SAML-D not on this machine")
def test_saml_d_real_values_all_mapped_or_counted():
    from src.data.public_sources import SAML_CURRENCY, SAML_FORMAT

    raw = pd.read_csv(REAL_SAML_D, usecols=["Payment_currency", "Received_currency", "Payment_type"])
    assert set(raw["Payment_type"].unique()) <= set(SAML_FORMAT)
    names = set(raw["Payment_currency"].unique()) | set(raw["Received_currency"].unique())
    print("SAML-D currencies with no IBM equivalent:", sorted(names - set(SAML_CURRENCY)))
    assert set(SAML_CURRENCY) <= names, "a mapping key is misspelled"


def test_amlsim_mapping(tmp_path):
    d = tmp_path / "amlsim"
    d.mkdir()
    pd.DataFrame({"TX_ID": [1, 2, 3], "SENDER_ACCOUNT_ID": [5, 6, 5], "RECEIVER_ACCOUNT_ID": [6, 7, 7],
                  "TX_TYPE": ["TRANSFER"] * 3, "TX_AMOUNT": [10.0, 20.5, 30.0], "TIMESTAMP": [2, 0, 1],
                  "IS_FRAUD": [True, False, False], "ALERT_ID": [4, -1, -1]}).to_csv(d / "transactions.csv", index=False)
    df = load_amlsim(d)
    assert list(df["ts"]) == [1640995200, 1640995200 + 86400, 1640995200 + 172800]  # time-sorted
    assert list(df["label"]) == [0, 0, 1]
    assert (df["from_bank"] == 0).all() and (df["to_bank"] == 0).all()
    assert (df["pay_format"] == PAYMENT_FORMATS.index("ACH")).all()
    assert (df["pay_currency"] == CURRENCIES.index("US Dollar")).all()
    assert (df["recv_currency"] == CURRENCIES.index("US Dollar")).all()
    assert df["amount_received"].iloc[2] == pytest.approx(10.0)


def test_account_hash_grouping(tmp_path):
    src = _write_saml(make_saml(), tmp_path / "s.csv")
    meta = build_dataset(src, tmp_path / "out", n_clients=3, source="saml_d", group_by="account_hash")
    assert meta["group_by"] == "account_hash" and meta["source_format"] == "saml_d"
    assert meta["bank_to_client"] == {} and meta["n_banks"] == 12
    seen = {}
    for c in range(3):
        acc = torch.load(tmp_path / "out" / f"client_{c}" / "table.pt", weights_only=True)["account"]
        for a in acc.unique().tolist():
            assert a not in seen, f"account {a} in clients {seen[a]} and {c}"
            seen[a] = c
    assert len(seen) == 12
    assert set(meta["clients"]) == {"0", "1", "2"}
    for c in meta["clients"].values():  # stats exist even for a client with no train fraud
        assert {"train", "val", "test"} <= set(c)


def test_ibm_build_unchanged(tiny_csv, tmp_path):
    # Hash recorded at 7f1d0f1, before build_dataset learned other sources.
    meta = build_dataset(tiny_csv, tmp_path / "out", n_clients=2)
    h = hashlib.sha256()
    for c in range(2):
        t = torch.load(tmp_path / "out" / f"client_{c}" / "table.pt", weights_only=True)
        for k in sorted(t):
            h.update(k.encode())
            h.update(t[k].numpy().tobytes())
    assert h.hexdigest() == "c00fbc3b5435216387c0c284b2df5d72f7a9c1908d5d121a7c86b069251d8710"
    assert meta["unknown_categories"] == {"pay_currency": 0, "recv_currency": 0, "pay_format": 0}
    assert meta["group_by"] == "bank" and meta["source_format"] == "ibm"
    on_disk = json.loads((tmp_path / "out" / "metadata" / "dataset_info.json").read_text())
    assert on_disk["unknown_categories"] == meta["unknown_categories"]
