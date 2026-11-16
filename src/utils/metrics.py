"""Fraud metrics that stay meaningful at ~0.1% positives."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def _both_classes(y_true) -> bool:
    return len(np.unique(y_true)) == 2


def recall_at_fpr(y_true, y_prob, fpr: float = 0.01):
    if not _both_classes(y_true):
        return None
    f, t, _ = roc_curve(y_true, y_prob, drop_intermediate=False)
    return float(t[f <= fpr].max())


def best_f1_threshold(y_true, y_prob) -> float:
    if not _both_classes(y_true):
        return 0.5
    p, r, thr = precision_recall_curve(y_true, y_prob)
    p, r = p[:-1], r[:-1]  # the last point has no threshold
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(thr[int(np.argmax(f1))])


ALERT_BUDGETS = (1000, 5000)


def alert_budget_metrics(y_true, y_prob) -> dict:
    """Precision and recall when only the top-K scores are alerted (K capped at n)."""
    order = np.argsort(-np.asarray(y_prob, dtype=np.float64), kind="stable")
    y = np.asarray(y_true).astype(int)
    out = {}
    for k in ALERT_BUDGETS:
        k_eff = min(k, len(y))
        hits = int(y[order[:k_eff]].sum())
        out[f"precision_at_{k}"] = hits / k_eff if k_eff else 0.0
        out[f"recall_at_{k}"] = hits / int(y.sum()) if y.sum() else None
    return out


def fraud_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    pred = (y_prob >= threshold).astype(int)
    both = _both_classes(y_true)
    return {
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else None,
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_prob, 0.01),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        **alert_budget_metrics(y_true, y_prob),
    }
