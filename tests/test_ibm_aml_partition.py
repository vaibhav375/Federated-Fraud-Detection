import numpy as np
import pandas as pd

from src.data.ibm_aml import assign_banks_to_clients, time_split


def test_greedy_assignment_exact():
    counts = pd.Series({1: 100, 2: 90, 3: 50, 4: 40, 5: 10})
    assert assign_banks_to_clients(counts, n_clients=2) == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}


def test_every_bank_assigned_once_and_fewer_banks_than_clients():
    counts = pd.Series({7: 5, 8: 3})
    assert assign_banks_to_clients(counts, n_clients=10) == {7: 0, 8: 1}


def test_greedy_ties_go_to_lower_bank_id_first():
    counts = pd.Series({9: 5, 3: 5})
    assert assign_banks_to_clients(counts, n_clients=2) == {3: 0, 9: 1}


def test_time_split_keeps_ties_together():
    ts = np.array([0, 1, 2, 3, 4, 5, 5, 5, 8, 9])
    assert time_split(ts).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
