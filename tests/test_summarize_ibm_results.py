import json

import experiments.summarize_ibm_results as summ


def _res(pr, priv=None, per_client=None):
    pooled = {"pr_auc": pr, "recall_at_1pct_fpr": 0.5, "f1": 0.4}
    return {"seeds": {"0": {"pooled": pooled, "per_client": per_client or {}, "privacy": priv,
                            "train_seconds": 10.0}},
            "mean": pooled, "std": {"pr_auc": 0.0, "recall_at_1pct_fpr": 0.0, "f1": 0.0}}


def test_table_shows_composed_and_account_epsilon_and_client_wins(tmp_path, capsys):
    pc = lambda a, b: {"0": {"pr_auc": a}, "1": {"pr_auc": b}}
    data = {"results": {
        "local": _res(0.1, per_client=pc(0.1, 0.5)),
        "fedprox": _res(0.3, per_client=pc(0.2, 0.4)),
        "client_dp_eps8.0": _res(0.2, priv={"epsilon_rdp_total": 13.46}),
        "account_dp_eps8.0": _res(0.1, priv={"0": {"epsilon_spent": 7.99}, "1": {"epsilon_spent": 8.0}}),
    }}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    summ.main(str(path))
    out = capsys.readouterr().out
    assert "13.5 (client-level, composed)" in out
    assert "max 8.00 (account-level)" in out
    assert "for [1] of 2 clients" in out
