import json

import pytest

from helpers import make_raw, write_raw
from src.data.ibm_aml import build_dataset

import experiments.run_ibm_comparison as runner


def _build(tiny_csv, tmp_path):
    build_dataset(tiny_csv, tmp_path / "data", n_clients=2)
    return str(tmp_path / "data")


def test_all_models_run_and_share_one_test_set(tiny_csv, tmp_path):
    data = _build(tiny_csv, tmp_path)
    out = runner.main(["--data-path", data, "--out", str(tmp_path / "r.json"),
                       "--rounds", "1", "--local-epochs", "1", "--batch-size", "16",
                       "--epsilons", "8", "--dp-batch-accounts", "4", "--delta", "1e-3", "--seeds", "0"])
    names = set(out["results"])
    assert names == {"local", "centralized", "gbdt", "fedavg", "fedprox",
                     "client_dp_eps8.0", "account_dp_eps8.0"}
    n_test = out["data"]["pooled"]["test"]["n"]
    for name in names:
        res = out["results"][name]["seeds"]["0"]
        assert res["pooled"]["n"] == n_test, name
        assert set(res["per_client"]) == {"0", "1"}, name
    priv = out["results"]["account_dp_eps8.0"]["seeds"]["0"]["privacy"]
    assert set(priv) == {"0", "1"}
    assert all(0 < v["epsilon_spent"] <= 8.0 for v in priv.values())
    assert out["results"]["client_dp_eps8.0"]["seeds"]["0"]["privacy"]["epsilon_rdp_total"] > 0
    assert out["results"]["fedprox"]["seeds"]["0"]["privacy"] is None
    assert json.loads((tmp_path / "r.json").read_text())["results"].keys() == out["results"].keys()


def test_aborts_when_delta_too_large_for_account_count(tiny_csv, tmp_path):
    data = _build(tiny_csv, tmp_path)
    with pytest.raises(SystemExit, match="delta"):
        runner.main(["--data-path", data, "--out", str(tmp_path / "r.json"),
                     "--models", "account_dp", "--rounds", "1", "--delta", "0.5"])


def test_mean_and_std_over_seeds(tiny_csv, tmp_path):
    data = _build(tiny_csv, tmp_path)
    out = runner.main(["--data-path", data, "--out", str(tmp_path / "r.json"), "--models", "gbdt",
                       "--seeds", "0", "1"])
    r = out["results"]["gbdt"]
    assert set(r["seeds"]) == {"0", "1"} and "pr_auc" in r["mean"] and "pr_auc" in r["std"]
