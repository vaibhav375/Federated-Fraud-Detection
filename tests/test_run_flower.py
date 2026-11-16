import json

import pytest

import experiments.run_flower as rf
import experiments.run_ibm_comparison as runner

TINY = ["--rounds", "2", "--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3"]


def _run(tiny3, *extra):
    out = tiny3 / "b1.json"
    rf.main(["--data-path", str(tiny3 / "data"), "--out", str(out), "--work-dir", str(tiny3 / "run"),
             *TINY, *extra])
    return json.loads(out.read_text())


def test_run_flower_writes_metrics_and_errors(tiny3):
    r = _run(tiny3, "--debug-dump-locals")
    assert r["secagg"] is True and r["debug_dump_locals"] is True and r["rounds"] == 2
    assert len(r["aggregation_errors"]) == 2 and max(r["aggregation_errors"]) <= r["error_bound"]
    assert len(r["round_seconds"]) == 2 and all(t > 0 for t in r["round_seconds"])
    assert r["metrics"]["pooled"]["pr_auc"] is not None


def test_dump_flag_recorded_and_off_by_default(tiny3):
    r = _run(tiny3, "--no-secagg")
    assert r["debug_dump_locals"] is False and r["aggregation_errors"] is None
    assert not (tiny3 / "run" / "locals").exists()


def test_flower_account_dp_reports_same_epsilon_as_plain_loop(tiny3):
    r = _run(tiny3, "--account-dp-eps", "8", "--no-secagg")
    plain = runner.main(["--data-path", str(tiny3 / "data"), "--out", str(tiny3 / "a0.json"),
                         "--models", "account_dp", "--epsilons", "8", *TINY])
    expected = plain["results"]["account_dp_eps8.0"]["seeds"]["0"]["privacy"]
    assert r["privacy"].keys() == expected.keys()
    for c, e in expected.items():
        assert r["privacy"][c]["epsilon_spent"] == pytest.approx(e["epsilon_spent"], abs=1e-9)
        assert r["privacy"][c]["sigma"] == pytest.approx(e["sigma"], abs=1e-12)
        assert r["privacy"][c]["steps"] == e["steps"]


def test_max_train_accounts_equals_form_is_respected():
    _, run_args = rf.parse(["--max-train-accounts=5000"])
    assert "20000" not in run_args and "--max-train-accounts=5000" in run_args


def test_account_dp_rejects_delta_too_large_before_training(tiny3, monkeypatch):
    monkeypatch.setattr(rf, "run", lambda *a, **k: pytest.fail("trained before checking delta"))
    with pytest.raises(SystemExit, match="delta"):
        _run(tiny3, "--account-dp-eps", "8", "--delta", "0.5")


HEAD = ["--local-epochs", "1", "--batch-size", "32", "--delta", "1e-3", "--dp-batch-accounts", "4", "--seeded-dp"]


def test_run_flower_head_dp_end_to_end(ckpt, tiny3, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"chosen": {"8.0": {"lr": 0.003, "layers": "classifier", "rounds": 2}}}))
    out = tmp_path / "head.json"
    rf.main(["--data-path", str(tiny3 / "data"), "--out", str(out), "--work-dir", str(tmp_path / "run"),
             "--ckpt", str(ckpt), "--settings", str(settings), "--account-dp-eps", "8",
             "--debug-dump-locals", *HEAD])
    r = json.loads(out.read_text())
    assert r["mode"] == "head_dp" and r["head_train"] == "classifier" and r["rounds"] == 2
    assert r["selection"] == "fixed_rounds" and r["dp_randomness"] == "seeded"
    assert len(r["aggregation_errors"]) == 2 and max(r["aggregation_errors"]) <= r["error_bound"]
    assert all(p["epsilon_spent"] <= 8.0 + 1e-6 for p in r["privacy"].values())
    assert r["metrics"]["pooled"]["pr_auc"] is not None


def test_run_flower_settings_missing_level_exits(ckpt, tiny3, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"chosen": {"8.0": {"lr": 0.003, "layers": "classifier", "rounds": 2}}}))
    with pytest.raises(SystemExit, match="3.0"):
        rf.main(["--data-path", str(tiny3 / "data"), "--out", str(tmp_path / "x.json"), "--work-dir",
                 str(tmp_path / "run"), "--ckpt", str(ckpt), "--settings", str(settings),
                 "--account-dp-eps", "3", *HEAD])
