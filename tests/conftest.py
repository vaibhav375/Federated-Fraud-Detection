import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import make_raw, write_raw  # noqa: E402


@pytest.fixture
def tiny_csv(tmp_path):
    return write_raw(make_raw(), tmp_path)


@pytest.fixture
def tiny_client(tmp_path):
    from src.data.ibm_aml import build_dataset
    from src.data.sequences import load_indexed_client

    build_dataset(write_raw(make_raw(), tmp_path), tmp_path / "out", n_clients=2)
    return load_indexed_client(tmp_path / "out", 0)


@pytest.fixture
def tiny3(tmp_path):
    """Three clients of unequal size (7 banks: 100/70/70 train examples)."""
    from src.data.ibm_aml import build_dataset

    build_dataset(write_raw(make_raw(n_banks=7), tmp_path), tmp_path / "data", n_clients=3)
    return tmp_path


@pytest.fixture
def ckpt(tmp_path):
    """A checkpoint pretrained for one epoch on a separate tiny public set."""
    from experiments.pretrain_public import main
    from src.data.ibm_aml import build_dataset

    (tmp_path / "pubraw").mkdir()
    build_dataset(write_raw(make_raw(seed=5), tmp_path / "pubraw"), tmp_path / "pub", n_clients=1)
    path = tmp_path / "pre.pt"
    main(["--data-path", str(tmp_path / "pub"), "--epochs", "1", "--out-ckpt", str(path),
          "--out-json", str(tmp_path / "pre.json"), "--batch-size", "32"])
    return path
