import numpy as np
import torch

from src.data.sequences import SequenceIndexDataset, build_history_index


def test_history_index_exact():
    out = build_history_index(np.array([7, 7, 3, 7, 3]), k=3)
    assert out.tolist() == [[-1, -1, 0], [-1, 0, 1], [-1, -1, 2], [0, 1, 3], [-1, 2, 4]]


def test_dataset_pads_and_flags():
    feats = torch.arange(10, dtype=torch.float32).reshape(5, 2)  # col 1 is pad_col
    feats[:, 1] = 0
    hist = torch.tensor(build_history_index(np.array([7, 7, 3, 7, 3]), k=3))
    ds = SequenceIndexDataset(feats, hist, torch.tensor([0, 0, 1, 0, 1]),
                              torch.tensor([7, 7, 3, 7, 3]), rows=torch.tensor([2, 3]), pad_col=1)
    seq, label = ds[0]  # table row 2: two padded steps then itself
    assert seq.tolist() == [[0, 1], [0, 1], [4, 0]] and label.item() == 1
    seqs, labels = ds.gather(torch.tensor([1]))
    assert seqs.shape == (1, 3, 2) and seqs[0, :, 0].tolist() == [0, 2, 6]
    assert ds.example_accounts.tolist() == [3, 7]
    assert ds.example_labels.tolist() == [1, 0] and len(ds) == 2
