import numpy as np
import torch

from helpers import make_raw, write_raw
from src.data.ibm_aml import (FEATURE_NAMES, NUM_FEATURES, PAD_COL, build_dataset,
                              encode_features, load_transactions)
from src.data.sequences import load_indexed_client


def test_features_are_causal(tiny_csv):
    df = load_transactions(tiny_csv)
    full = encode_features(df)
    for j in (0, 57, 399):
        assert np.array_equal(encode_features(df.iloc[: j + 1])[: j + 1], full[: j + 1])


def test_features_shape_and_range(tiny_csv):
    x = encode_features(load_transactions(tiny_csv))
    assert x.shape == (400, NUM_FEATURES) and x.dtype == np.float32
    assert len(FEATURE_NAMES) == NUM_FEATURES and FEATURE_NAMES[PAD_COL] == "is_pad"
    assert np.isfinite(x).all() and np.abs(x).max() <= 5.0


def test_account_history_features_count_only_earlier_rows(tmp_path):
    raw = make_raw(n_rows=3, n_banks=1, accounts_per_bank=1, n_fraud=0)
    raw["Account.1"] = ["R1", "R2", "R1"]
    raw["To Bank"] = "001"
    x = encode_features(load_transactions(write_raw(raw, tmp_path)))
    names = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert np.allclose(x[:, names["acct_tx_count_log"]], np.log1p([0, 1, 2]) / 10)
    assert np.allclose(x[:, names["acct_distinct_receivers_log"]], np.log1p([0, 1, 2]) / 10)
    assert np.allclose(x[:, names["secs_since_prev_log"]], [0, np.log1p(60) / 15, np.log1p(60) / 15])


def test_unknown_format_gives_zero_one_hot(tmp_path):
    raw = make_raw(n_rows=4, n_fraud=0)
    raw.loc[0, "Payment Format"] = "Carrier Pigeon"
    x = encode_features(load_transactions(write_raw(raw, tmp_path)))
    assert x[0, 13:20].sum() == 0 and x[1, 13:20].sum() == 1


def test_build_invariants(tiny_csv, tmp_path):
    meta = build_dataset(tiny_csv, tmp_path / "out", n_clients=3)
    tables = [torch.load(tmp_path / "out" / f"client_{i}" / "table.pt") for i in range(3)]
    assert sum(len(t["labels"]) for t in tables) == 400
    assert sum(int(t["labels"].sum()) for t in tables) == 12
    accounts = [set(t["account"].tolist()) for t in tables]
    assert not (accounts[0] & accounts[1]) and not (accounts[1] & accounts[2])
    for t in tables:
        assert (t["ts"][1:] >= t["ts"][:-1]).all()
        assert (t["split"][1:] >= t["split"][:-1]).all()  # chronological
        h = t["history"]
        j = torch.arange(len(h)).unsqueeze(1)
        valid = h >= 0
        assert (h[valid] <= j.expand_as(h)[valid]).all()  # never looks ahead
        assert (t["account"][h.clamp(min=0)] == t["account"].unsqueeze(1))[valid].all()
    assert meta["format"] == "indexed_v1" and meta["num_features"] == 35
    assert "fraud_rate" in meta["clients"]["0"]["test"]
    assert meta["fraud_rate_warning"] is False


def test_fraud_concentrated_in_test_period_sets_warning(tmp_path):
    raw = make_raw()
    raw["Is Laundering"] = 0
    raw.loc[[10, 390, 391, 392, 393, 394, 395, 396], "Is Laundering"] = 1
    raw.loc[250, "Is Laundering"] = 1
    meta = build_dataset(write_raw(raw, tmp_path), tmp_path / "out", n_clients=2)
    assert meta["fraud_rate_warning"] is True


def test_max_train_accounts_keeps_whole_accounts(tiny_csv, tmp_path):
    build_dataset(tiny_csv, tmp_path / "out", n_clients=2)
    c = load_indexed_client(tmp_path / "out", 0, max_train_accounts=2, seed=0)
    assert len(set(c["train"].example_accounts.tolist())) == 2 == c["n_train_accounts"]
    full = load_indexed_client(tmp_path / "out", 0)
    assert len(c["test"]) == len(full["test"]) and c["num_features"] == 35
