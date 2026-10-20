from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from dashcam_risk.leakage import assert_disjoint_clip_ids
from dashcam_risk.schema import REQUIRED_COLUMNS


# Slices below this many clips are flagged unreliable (reported, not hidden).
SMALL_SLICE_FLOOR = 30


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df["clip_id"] = df["clip_id"].astype(str)
    return df


def _two_way_clip_split(
    clips: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = clips[stratify_col] if stratify_col in clips.columns else None
    a, b = train_test_split(clips, test_size=test_size, random_state=seed, stratify=strat)
    return a.copy(), b.copy()


def train_val_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    stratify_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way clip-level split (used to carve a validation set inside holdout mode).

    Stratification is best-effort: on tiny or heavily imbalanced slices sklearn
    cannot allocate every class to both sides, so we fall back to an
    unstratified split. Composition is still disclosed via split_counts.
    """
    clips = df.drop_duplicates("clip_id").copy()
    try:
        train, val = _two_way_clip_split(clips, val_size, seed, stratify_col)
    except ValueError:
        train, val = _two_way_clip_split(clips, val_size, seed, stratify_col=None)
    assert_disjoint_clip_ids(train["clip_id"], val["clip_id"])
    return train, val


def clip_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.2,
    stratify_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Split by clip_id. Optional stratify on label."""
    clips = df.drop_duplicates("clip_id").copy()
    trainval, test = _two_way_clip_split(clips, test_size, seed, stratify_col)
    rel_val = val_size / max(1.0 - test_size, 1e-6)
    train, val = _two_way_clip_split(trainval, rel_val, seed, stratify_col)
    assert_disjoint_clip_ids(train["clip_id"], val["clip_id"], test["clip_id"])
    keys = {
        "train": set(train["clip_id"]),
        "val": set(val["clip_id"]),
        "test": set(test["clip_id"]),
    }
    out = {}
    for name, ids in keys.items():
        out[name] = df[df["clip_id"].isin(ids)].copy()
        out[name]["split"] = name
    return out


def holdout_by_column(
    df: pd.DataFrame,
    column: str,
    values: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domain-shift split: train on everything except listed values of a slice."""
    mask = df[column].astype(str).isin([str(v) for v in values])
    test = df[mask].copy()
    train = df[~mask].copy()
    assert_disjoint_clip_ids(train["clip_id"], test["clip_id"])
    return train, test


def slice_warning(n_clips: int, floor: int = SMALL_SLICE_FLOOR) -> str | None:
    """Flag slices too small to trust; the flag is reported, never silently dropped."""
    if n_clips < floor:
        return (
            f"holdout slice has {n_clips} clips (< {floor}); "
            f"treat its metrics as unreliable"
        )
    return None


def split_counts(parts: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for name, part in parts.items():
        summary[name] = {
            "clips": part["clip_id"].nunique(),
            "rows": len(part),
        }
        if "label" in part.columns:
            vc = part.drop_duplicates("clip_id")["label"].value_counts().to_dict()
            summary[name].update({f"label_{k}": int(v) for k, v in vc.items()})
    return summary
