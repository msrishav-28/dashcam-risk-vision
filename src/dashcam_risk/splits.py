from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from dashcam_risk.leakage import assert_disjoint_clip_ids
from dashcam_risk.schema import REQUIRED_COLUMNS


def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df["clip_id"] = df["clip_id"].astype(str)
    return df


def clip_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.2,
    stratify_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Split by clip_id. Optional stratify on label."""
    clips = df.drop_duplicates("clip_id").copy()
    strat = clips[stratify_col] if stratify_col in clips.columns else None
    trainval, test = train_test_split(
        clips,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )
    strat_tv = trainval[stratify_col] if stratify_col in trainval.columns else None
    rel_val = val_size / max(1.0 - test_size, 1e-6)
    train, val = train_test_split(
        trainval,
        test_size=rel_val,
        random_state=seed,
        stratify=strat_tv,
    )
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
