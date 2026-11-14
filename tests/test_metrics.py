import numpy as np
import pytest

from src.utils.metrics import best_f1_threshold, fraud_metrics, recall_at_fpr

Y = np.array([0, 0, 0, 0, 1, 1])
P = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])


def test_recall_at_fpr_exact():
    assert recall_at_fpr(Y, P, fpr=0.0) == 0.5
    assert recall_at_fpr(Y, P, fpr=0.25) == 1.0


def test_best_f1_threshold_exact():
    assert best_f1_threshold(Y, P) == pytest.approx(0.8)
    m = fraud_metrics(Y, P, threshold=0.8)
    assert m["f1"] == pytest.approx(0.8) and m["n"] == 6 and m["n_fraud"] == 2
    assert m["threshold"] == pytest.approx(0.8)


def test_single_class_returns_none_not_error():
    m = fraud_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5))
    assert m["pr_auc"] is None and m["roc_auc"] is None and m["recall_at_1pct_fpr"] is None
    assert m["f1"] == 0.0


def test_best_f1_threshold_with_no_fraud_is_finite():
    assert np.isfinite(best_f1_threshold(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4])))
