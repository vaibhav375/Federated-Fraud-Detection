import numpy as np
import pandas as pd
import pytest

from experiments.analyze_client0 import main, per_bank_report


def test_per_bank_report():
    df = pd.DataFrame({"from_bank": [3, 3, 3, 3, 5, 5, 5, 9], "label": [0, 1, 0, 1, 0, 0, 1, 0]})
    split = np.array([0, 0, 2, 2, 2, 2, 0, 2])
    probs = np.array([0.9, 0.1, 0.3, 0.2, 0.4])  # test rows in order: bank 3 x2, bank 5 x2, bank 9
    rep = per_bank_report(df, split, probs)
    assert [r["bank"] for r in rep] == [3, 5, 9]  # by test_n, then bank
    b3, b5, b9 = rep
    assert (b3["train_n"], b3["train_fraud"], b3["test_n"], b3["test_fraud"]) == (2, 1, 2, 1)
    assert b3["test_pr_auc"] == pytest.approx(0.5)  # the fraud row scored below the legit one
    assert b5["test_pr_auc"] is None and b9["test_pr_auc"] is None  # no fraud in their test rows
    assert sum(r["test_n"] for r in rep) == 5
    assert sum(r["share_of_client"] for r in rep) == pytest.approx(1.0)
    assert b3["share_of_client"] == pytest.approx(0.5)


def test_alignment_check(tiny_csv, tmp_path):
    scores = tmp_path / "s.npz"
    np.savez(scores, client_0=np.zeros(3))
    with pytest.raises(ValueError, match="scores"):
        main(["--raw", str(tiny_csv), "--scores", str(scores), "--client", "0", "--n-clients", "2",
              "--out", str(tmp_path / "c0.json")])


def test_end_to_end(tiny_csv, tmp_path):
    import json

    from src.data.ibm_aml import build_dataset
    from src.data.sequences import load_indexed_client

    build_dataset(tiny_csv, tmp_path / "d", n_clients=2)
    n_test = len(load_indexed_client(tmp_path / "d", 0)["test"])
    np.savez(tmp_path / "s.npz", client_0=np.linspace(0, 1, n_test))
    out = main(["--raw", str(tiny_csv), "--scores", str(tmp_path / "s.npz"), "--client", "0",
                "--n-clients", "2", "--out", str(tmp_path / "c0.json")])
    assert sum(r["test_n"] for r in out["banks"]) == n_test
    assert json.loads((tmp_path / "c0.json").read_text())["client"] == 0
    assert 0 < out["largest_bank_share"] <= 1
