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
