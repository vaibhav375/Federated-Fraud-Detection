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
