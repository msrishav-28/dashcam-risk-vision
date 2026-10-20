import pytest

from dashcam_risk.leakage import (
    LeakageError,
    assert_disjoint_clip_ids,
    assert_input_before_alert,
    assert_uniform_length,
    sample_times,
)
from dashcam_risk.schema import ClipRecord


def test_disjoint_ok():
    assert_disjoint_clip_ids(["a", "b"], ["c"])


def test_disjoint_fails():
    with pytest.raises(LeakageError):
        assert_disjoint_clip_ids(["a", "b"], ["b"])


def test_alert_cutoff():
    rec = ClipRecord(
        clip_id="x",
        path="x.mp4",
        label="risk",
        alert_time_s=10.0,
        event_time_s=12.0,
        duration_s=40.0,
    )
    assert_input_before_alert(9.5, rec)
    with pytest.raises(LeakageError):
        assert_input_before_alert(10.0, rec)


def test_alert_cutoff_applies_to_normal_clip_with_alert():
    # docs/LEAKAGE.md: every example stays before alert_time_s *when alert exists*,
    # regardless of label. Normal clips carrying an alert_time used to slip past.
    rec = ClipRecord(
        clip_id="n",
        path="n.mp4",
        label="normal",
        alert_time_s=10.0,
        duration_s=40.0,
    )
    assert_input_before_alert(9.5, rec)
    with pytest.raises(LeakageError):
        assert_input_before_alert(10.0, rec)


def test_normal_clip_without_alert_has_no_time_restriction():
    # Without an alert_time the full clip is legal input; duration is not a cutoff.
    rec = ClipRecord(clip_id="n", path="n.mp4", label="normal", duration_s=40.0)
    assert_input_before_alert(39.999, rec)


def test_risk_clip_event_time_fallback_still_enforced():
    # Risk clip without alert_time: event_time_s is the last legal input time.
    rec = ClipRecord(
        clip_id="x",
        path="x.mp4",
        label="risk",
        event_time_s=12.0,
        duration_s=40.0,
    )
    assert_input_before_alert(11.5, rec)
    with pytest.raises(LeakageError):
        assert_input_before_alert(12.0, rec)


def test_sample_times_stay_before_cutoff():
    times = sample_times(duration_s=40, cutoff_s=8, n_frames=8)
    assert len(times) == 8
    assert times[-1] < 8
    assert times[0] >= 0


def test_uniform_length():
    assert_uniform_length([8, 8, 8])
    with pytest.raises(LeakageError):
        assert_uniform_length([8, 16])
