import json

import pytest
import torch

from helpers import make_raw, write_raw


@pytest.fixture
def public_dir(tmp_path):
    from src.data.ibm_aml import build_dataset

    build_dataset(write_raw(make_raw(), tmp_path), tmp_path / "pub", n_clients=1)
    return tmp_path / "pub"


def _args(public_dir):
    from experiments.run_ibm_comparison import parse_args

    return parse_args(["--data-path", str(public_dir), "--batch-size", "32"])


def test_pretrain_returns_best_epoch_by_val(public_dir):
    from experiments.pretrain_public import pretrain
    from src.models.temporal_graph_transformer import TemporalGraphTransformer

    out = pretrain(public_dir, epochs=2, seed=0, args=_args(public_dir))
    vals = out["val_pr_auc_by_epoch"]
    assert len(vals) == 2
    assert out["best_epoch"] == max(range(2), key=lambda i: vals[i]) + 1
    assert out["state_dict"].keys() == TemporalGraphTransformer(out["config"]).state_dict().keys()
    assert out["num_features"] == 35
    assert out["test"]["pr_auc"] is not None


def test_cli_writes_checkpoint_and_json(public_dir, tmp_path):
    from experiments.pretrain_public import main

    ckpt, js = tmp_path / "ck" / "p.pt", tmp_path / "res" / "p.json"
    main(["--data-path", str(public_dir), "--epochs", "1", "--out-ckpt", str(ckpt), "--out-json", str(js),
          "--batch-size", "32"])
    saved = torch.load(ckpt, weights_only=False)
    assert saved["num_features"] == 35 and set(saved) == {"state_dict", "config", "num_features", "source"}
    report = json.loads(js.read_text())
    assert "state_dict" not in report and report["best_epoch"] == 1
