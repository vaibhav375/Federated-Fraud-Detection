import os
from pathlib import Path

import pandas as pd
import pytest

from src.data.ibm_aml import CURRENCIES, PAYMENT_FORMATS, RAW_COLUMNS

RAW = Path(os.environ.get("IBM_AML_RAW",
                          "/Volumes/Untitled/datasets_capstone/ibm_aml/hi_small/HI-Small_Trans.csv"))


@pytest.mark.skipif(not RAW.exists(), reason="raw IBM file not downloaded")
def test_real_file_matches_assumed_schema():
    head = pd.read_csv(RAW, nrows=200_000)
    assert list(head.columns) == RAW_COLUMNS
    assert set(head["Payment Format"]) <= set(PAYMENT_FORMATS)
    assert set(head["Payment Currency"]) | set(head["Receiving Currency"]) <= set(CURRENCIES)
    assert set(head["Is Laundering"]) <= {0, 1}
    pd.to_datetime(head["Timestamp"], format="%Y/%m/%d %H:%M")
