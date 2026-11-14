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


def test_deep_models_freeze_heads_whose_loss_weight_is_zero():
    # Adam's weight decay alone walks the unused contrastive temperature to 0,
    # then sim/T overflows and 0 * inf = NaN poisons every gradient (seen at batch 788).
    import copy
    from run_training import DEFAULT_CONFIG
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["dataset"]["num_features"] = 35
    m = runner.model_builder(cfg)()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and not any(n.startswith(("contrastive_head", "aux_head")) for n in trainable)


def test_aggregation_does_not_release_local_memory_banks():
    # train_local fills each local contrastive memory bank with projections of real
    # training rows; clip_update/add_server_noise only touch parameters, so averaged
    # buffers would leave the bank outside the DP mechanism.
    import copy
    import torch
    from run_training import DEFAULT_CONFIG
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["dataset"]["num_features"] = 35
    build = runner.model_builder(cfg)
    global_model = build()
    locals_ = []
    for _ in range(2):
        m = build()
        m.load_state_dict(global_model.state_dict())
        m.contrastive_head.memory_bank.add_(1.0)
        m.contrastive_head.memory_ptr.add_(5)
        locals_.append(m)
    out = runner.aggregate(global_model, locals_, [3, 1], build)
    for name, buf in global_model.contrastive_head.named_buffers():
        assert torch.equal(dict(out.contrastive_head.named_buffers())[name], buf), name


def test_local_fit_extraction_preserves_federated_results(tiny_csv, tmp_path):
    # Golden values recorded at 8835099, before local_fit existed; the A0 JSON must stay reproducible.
    r = runner.main(["--data-path", _build(tiny_csv, tmp_path), "--out", str(tmp_path / "r.json"),
                     "--models", "fedprox", "account_dp", "--epsilons", "8", "--rounds", "2",
                     "--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3"])["results"]
    fp, dp = r["fedprox"]["seeds"]["0"], r["account_dp_eps8.0"]["seeds"]["0"]
    assert fp["pooled"]["pr_auc"] == pytest.approx(0.06821378340365682, abs=1e-12)
    assert fp["pooled"]["roc_auc"] == pytest.approx(0.4551282051282051, abs=1e-12)
    assert fp["pooled"]["threshold"] == pytest.approx(0.039456289261579514, abs=1e-12)
    assert dp["pooled"]["pr_auc"] == pytest.approx(0.06805555555555555, abs=1e-12)
    assert dp["pooled"]["roc_auc"] == pytest.approx(0.44871794871794873, abs=1e-12)
    assert dp["pooled"]["threshold"] == pytest.approx(0.03749164193868637, abs=1e-12)
    assert dp["privacy"]["0"]["epsilon_spent"] == pytest.approx(7.991569684852897, abs=1e-12)


def test_local_fit_trains_in_place(tiny_client):
    import torch
    from helpers import small_model_for
    m = small_model_for(tiny_client)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    args = runner.parse_args(["--local-epochs", "1", "--batch-size", "32"])
    runner.local_fit(m, tiny_client, args, [p.detach().clone() for p in m.parameters()],
                     torch.Generator().manual_seed(0), mu=0.1)
    changed = [n for n, p in m.named_parameters() if p.requires_grad and not torch.equal(p, before[n])]
    assert changed
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters() if not p.requires_grad)
