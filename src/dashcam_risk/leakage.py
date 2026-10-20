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


def assert_input_before_alert(
    t_last_input_s: float,
    record: ClipRecord,
    eps: float = 1e-3,
) -> None:
    cutoff = record.cutoff_s()
    if cutoff is None:
        return
    if t_last_input_s >= cutoff - eps and record.is_risk:
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
