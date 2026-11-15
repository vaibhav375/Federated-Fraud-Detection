import pandas as pd
import pytest

from helpers import make_raw, write_raw


def _two(tmp_path, public, private):
    (tmp_path / "pub").mkdir()
    (tmp_path / "priv").mkdir()
    return write_raw(public, tmp_path / "pub"), write_raw(private, tmp_path / "priv")


def _renamed(df):
    out = df.copy()
    out["Account"] = "X" + out["Account"]
    out["Account.1"] = "X" + out["Account.1"]
    return out


def test_disjoint_files_report_zero_overlap(tmp_path):
    from experiments.check_public_overlap import overlap_report

    pub, priv = _two(tmp_path, make_raw(), _renamed(make_raw(seed=1)))
    r = overlap_report(pub, priv)
    assert r["shared_rows"] == 0 and r["shared_accounts"] == 0
    assert r["public_rows"] == 400 and r["private_rows"] == 400
    assert r["public_accounts"] == 30 and r["private_accounts"] == 30


def test_copied_rows_are_counted(tmp_path):
    from experiments.check_public_overlap import overlap_report

    public = make_raw()
    private = pd.concat([public, _renamed(make_raw(n_rows=5, seed=2))], ignore_index=True)
    pub, priv = _two(tmp_path, public, private)
    r = overlap_report(pub, priv)
    assert r["shared_rows"] == len(public)
    assert r["shared_accounts"] == 30


def test_cli_exits_nonzero_on_heavy_overlap(tmp_path):
    from experiments.check_public_overlap import main

    public = make_raw()
    pub, priv = _two(tmp_path, public, public)
    with pytest.raises(SystemExit) as e:
        main([str(pub), str(priv)])
    assert e.value.code == 1


def test_cli_passes_on_disjoint_files(tmp_path):
    from experiments.check_public_overlap import main

    pub, priv = _two(tmp_path, make_raw(), _renamed(make_raw(seed=1)))
    assert main([str(pub), str(priv)])["shared_rows"] == 0
