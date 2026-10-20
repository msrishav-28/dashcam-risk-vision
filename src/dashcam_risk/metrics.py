from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out = {
        "n": float(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
        "auprc": float("nan"),
        "auroc": float("nan"),
    }
    if y_true.min() != y_true.max():
        out["auprc"] = float(average_precision_score(y_true, y_score))
        out["auroc"] = float(roc_auc_score(y_true, y_score))
    return out


def lead_time_recall(
    is_risk: np.ndarray,
    seconds_to_event: np.ndarray,
    score: np.ndarray,
    threshold: float,
    horizons_s: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> dict[str, float]:
    """Share of risk clips with score>=threshold and predicted lead >= horizon."""
    is_risk = np.asarray(is_risk).astype(bool)
    lead = np.asarray(seconds_to_event).astype(float)
    score = np.asarray(score).astype(float)
    out: dict[str, float] = {}
    risk_n = max(int(is_risk.sum()), 1)
    for h in horizons_s:
        ok = is_risk & (score >= threshold) & (lead >= h)
        out[f"recall_lead_ge_{h:.1f}s"] = float(ok.sum() / risk_n)
    return out


def time_to_event_mae(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> float:
    y_true_s = np.asarray(y_true_s).astype(float)
    y_pred_s = np.asarray(y_pred_s).astype(float)
    if len(y_true_s) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_s - y_pred_s)))
