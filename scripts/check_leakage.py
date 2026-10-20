#!/usr/bin/env python3
"""Fail the build if a manifest split leaks clip_ids or input times."""
from __future__ import annotations

import argparse

from dashcam_risk.dataset import row_to_record
from dashcam_risk.leakage import assert_disjoint_clip_ids, assert_input_before_alert, sample_times
from dashcam_risk.splits import clip_level_split, load_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/example_manifest.csv")
    p.add_argument("--n-frames", type=int, default=8)
    args = p.parse_args()
    df = load_manifest(args.manifest)
    parts = clip_level_split(df)
    assert_disjoint_clip_ids(*(part["clip_id"] for part in parts.values()))
    for _, row in df.iterrows():
        rec = row_to_record(row)
        duration = rec.duration_s or 40.0
        times = sample_times(duration, rec.cutoff_s(), args.n_frames)
        assert_input_before_alert(times[-1], rec)
    print("leakage checks passed", {k: v["clip_id"].nunique() for k, v in parts.items()})


if __name__ == "__main__":
    main()
