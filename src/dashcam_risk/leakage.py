from __future__ import annotations

from typing import Iterable, Sequence

from dashcam_risk.schema import ClipRecord


class LeakageError(ValueError):
    pass


def assert_disjoint_clip_ids(*groups: Iterable[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        ids = set(group)
        overlap = seen & ids
        if overlap:
            raise LeakageError(f"clip_id leakage across splits: {sorted(overlap)[:8]}")
        seen |= ids


def restricted_cutoff_s(record: ClipRecord) -> float | None:
    """Cutoff that restricts model input, or None when the whole clip is legal.

    An alert_time_s restricts every clip that carries one (docs/LEAKAGE.md:
    "t_last_input_s < alert_time_s when alert exists", any label). A risk clip
    without an alert_time falls back to its event_time_s. A normal clip's
    duration is a sampling window, not a restriction.
    """
    if record.alert_time_s is not None:
        return float(record.alert_time_s)
    if record.is_risk and record.event_time_s is not None:
        return float(record.event_time_s)
    return None


def assert_input_before_alert(
    t_last_input_s: float,
    record: ClipRecord,
    eps: float = 1e-3,
) -> None:
    cutoff = restricted_cutoff_s(record)
    if cutoff is None:
        return
    # strictly before: a frame sampled at exactly the cutoff has already seen
    # the onset of the event, so `>=` is rejected, not only `>`
    if t_last_input_s >= cutoff - eps:
        raise LeakageError(
            f"{record.clip_id}: last input {t_last_input_s:.3f}s is not before "
            f"cutoff {cutoff:.3f}s"
        )


def assert_uniform_length(lengths: Sequence[int]) -> None:
    if not lengths:
        return
    if len(set(lengths)) != 1:
        raise LeakageError(
            f"variable sequence length can leak label: unique lengths={sorted(set(lengths))}"
        )


def sample_times(
    duration_s: float,
    cutoff_s: float | None,
    n_frames: int,
) -> list[float]:
    """Evenly sample n_frames in [0, usable) so length cannot encode the label."""
    usable = duration_s if cutoff_s is None else min(duration_s, cutoff_s)
    usable = max(usable, 1e-3)
    if n_frames == 1:
        return [max(usable * 0.5, 0.0)]
    step = usable / n_frames
    # stay strictly inside the usable window
    return [min((i + 0.5) * step, usable - 1e-3) for i in range(n_frames)]
