import numpy as np
import pytest

from dashcam_risk.metrics import classification_report, lead_time_recall


def test_auprc_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    r = classification_report(y, p)
    assert r["auprc"] > 0.99


def test_lead_time():
    is_risk = np.array([1, 1, 1, 0])
    lead = np.array([2.5, 1.2, 0.4, -1])
    score = np.array([0.9, 0.9, 0.9, 0.1])
    out = lead_time_recall(is_risk, lead, score, threshold=0.5)
    assert out["recall_lead_ge_2.0s"] == pytest.approx(1 / 3)
    assert out["recall_lead_ge_1.0s"] == pytest.approx(2 / 3)
