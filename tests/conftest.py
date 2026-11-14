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
