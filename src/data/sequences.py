"""Per-entity transaction histories, gathered lazily so sequences are never
materialized as one dense (n, k, F) tensor."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_history_index(entity: np.ndarray, k: int = 10) -> np.ndarray:
    """Row j -> indices of the k most recent rows with the same entity, ending
    with j, left-padded with -1. Rows must already be in time order."""
    entity = np.asarray(entity)
    n = len(entity)
    order = np.argsort(entity, kind="stable")  # stable keeps time order inside a group
    sorted_ent = entity[order]
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = sorted_ent[1:] != sorted_ent[:-1]
    group_start = np.maximum.accumulate(np.where(is_start, np.arange(n), 0))
    out = np.full((n, k), -1, dtype=np.int64)
    pos = np.arange(n)
    for m in range(k):
        back = pos - m
        valid = back >= group_start
        out[order[valid], k - 1 - m] = order[back[valid]]
    return out


class SequenceIndexDataset(Dataset):
    def __init__(self, features, history, labels, account, rows, pad_col):
        self.features = features
        self.history = history
        self.labels = labels
        self.account = account
        self.rows = rows
        self.pad_col = pad_col

    def __len__(self):
        return len(self.rows)

    def gather(self, idx: torch.Tensor):
        r = self.rows[idx]
        h = self.history[r]
        pad = h < 0
        seqs = self.features[h.clamp(min=0)]
        seqs[pad] = 0.0
        seqs[..., self.pad_col][pad] = 1.0
        return seqs, self.labels[r]

    def __getitem__(self, i):
        seqs, labels = self.gather(torch.tensor([i]))
        return seqs[0], labels[0]

    @property
    def example_labels(self) -> torch.Tensor:
        return self.labels[self.rows]

    @property
    def example_accounts(self) -> torch.Tensor:
        return self.account[self.rows]


def load_indexed_client(data_dir: Path, client_id: int, max_train_accounts=None, seed: int = 0) -> dict:
    t = torch.load(Path(data_dir) / f"client_{client_id}" / "table.pt", weights_only=True)
    from src.data.ibm_aml import PAD_COL

    def make(rows):
        return SequenceIndexDataset(t["features"], t["history"], t["labels"], t["account"], rows, PAD_COL)

    rows = {s: (t["split"] == i).nonzero(as_tuple=True)[0] for i, s in enumerate(("train", "val", "test"))}
    train_accounts = torch.unique(t["account"][rows["train"]])
    if max_train_accounts is not None and len(train_accounts) > max_train_accounts:
        g = torch.Generator().manual_seed(seed)
        keep = train_accounts[torch.randperm(len(train_accounts), generator=g)[:max_train_accounts]]
        rows["train"] = rows["train"][torch.isin(t["account"][rows["train"]], keep)]
        train_accounts = keep
    out = {s: make(r) for s, r in rows.items()}
    out["num_features"] = int(t["features"].shape[1])
    out["n_train_accounts"] = int(len(train_accounts))
    return out
