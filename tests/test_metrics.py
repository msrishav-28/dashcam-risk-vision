import numpy as np
import pytest

from dashcam_risk.metrics import (
    aggregate_seed_runs,
    choose_threshold_f1,
    classification_report,
    lead_time_recall,
    time_to_event_mae,
)


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


def test_time_to_event_mae():
    assert time_to_event_mae(np.array([1.0, 3.0]), np.array([2.0, 1.0])) == pytest.approx(1.5)
    assert np.isnan(time_to_event_mae(np.array([]), np.array([])))


def test_choose_threshold_f1_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t = choose_threshold_f1(y, p)
    assert 0.2 < t < 0.8
    pred = (p >= t).astype(int)
    assert pred.tolist() == y.tolist()


def test_choose_threshold_f1_degenerate():
    y = np.array([1, 1, 1])
    assert choose_threshold_f1(y, np.array([0.3, 0.5, 0.9])) == 0.5
    assert choose_threshold_f1(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_aggregate_seed_runs():
    agg = aggregate_seed_runs([{"auprc": 0.5, "model": "x"}, {"auprc": 1.0, "model": "x"}])
    assert agg["n_seeds"] == 2
    assert agg["mean"]["auprc"] == pytest.approx(0.75)
    assert agg["std"]["auprc"] == pytest.approx(0.25)
    assert "model" not in agg["mean"]
    # key missing in one seed: not aggregatable
    agg2 = aggregate_seed_runs([{"auprc": 0.5}, {"auprc": 1.0, "extra": 2.0}])
    assert "extra" not in agg2["mean"]
