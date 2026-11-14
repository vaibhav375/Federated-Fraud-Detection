import numpy as np

from helpers import make_raw, write_raw
from src.data.ibm_aml import load_transactions


def test_load_columns_and_dtypes(tiny_csv):
    df = load_transactions(tiny_csv)
    assert list(df.columns) == ["ts", "from_bank", "to_bank", "from_acct", "to_acct",
                                "amount_paid", "amount_received", "pay_currency",
                                "recv_currency", "pay_format", "label"]
    assert len(df) == 400 and int(df["label"].sum()) == 12
    assert df["ts"].is_monotonic_increasing
    assert df["amount_paid"].dtype == np.float32


def test_same_account_string_at_two_banks_gets_two_codes(tmp_path):
    raw = make_raw(n_rows=4, n_banks=2, accounts_per_bank=1, n_fraud=0)
    raw["Account"] = "A1"
    raw["From Bank"] = ["001", "002", "001", "002"]
    df = load_transactions(write_raw(raw, tmp_path))
    assert df["from_acct"].nunique() == 2


def test_unknown_category_is_minus_one(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0)
    raw.loc[0, "Payment Format"] = "Carrier Pigeon"
    assert load_transactions(write_raw(raw, tmp_path))["pay_format"].iloc[0] == -1


def test_tied_timestamps_keep_file_order(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0)
    raw["Timestamp"] = "2022/09/01 00:00"
    raw["Amount Paid"] = [1.0, 2.0, 3.0, 4.0]
    assert load_transactions(write_raw(raw, tmp_path))["amount_paid"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_zero_padded_bank_codes_stay_distinct(tmp_path):
    raw = make_raw(n_rows=2, n_fraud=0)
    raw["From Bank"] = ["010", "10"]
    raw["Account"] = "X"
    df = load_transactions(write_raw(raw, tmp_path))
    assert df["from_bank"].nunique() == 2 and df["from_acct"].nunique() == 2
