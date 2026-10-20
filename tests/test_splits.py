import pandas as pd

from dashcam_risk.splits import (
    clip_level_split,
    holdout_by_column,
    slice_warning,
    train_val_split,
)


def _toy():
    rows = []
    for i in range(20):
        rows.append(
            {
                "clip_id": f"c{i}",
                "path": f"c{i}.pt",
                "label": "risk" if i < 8 else "normal",
                "time_of_day": "night" if i % 5 == 0 else "day",
                "duration_s": 40,
                "alert_time_s": 12 if i < 8 else None,
            }
        )
    return pd.DataFrame(rows)


def test_clip_split_disjoint():
    parts = clip_level_split(_toy(), seed=0)
    ids = {k: set(v["clip_id"]) for k, v in parts.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert sum(len(v) for v in ids.values()) == 20


def test_night_holdout():
    train, test = holdout_by_column(_toy(), "time_of_day", ["night"])
    assert set(test["time_of_day"]) == {"night"}
    assert "night" not in set(train["time_of_day"])


def test_train_val_split_disjoint():
    train, val = train_val_split(_toy(), seed=0, val_size=0.25)
    ids_train, ids_val = set(train["clip_id"]), set(val["clip_id"])
    assert ids_train.isdisjoint(ids_val)
    assert len(ids_train) + len(ids_val) == 20


def test_slice_warning_flags_small_slices():
    assert slice_warning(4, floor=30) is not None
    assert "4 clips" in slice_warning(4, floor=30)
    assert slice_warning(30, floor=30) is None
