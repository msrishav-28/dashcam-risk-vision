#!/usr/bin/env python3
"""Template: turn a source-specific CSV into the canonical manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CANON = [
    "clip_id",
    "trip_id",
    "path",
    "label",
    "event_type",
    "weather",
    "time_of_day",
    "scene",
    "duration_s",
    "fps",
    "event_time_s",
    "alert_time_s",
    "license",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="source annotation table")
    p.add_argument("--out", default="data/processed/manifest.csv")
    args = p.parse_args()
    raw = pd.read_csv(args.input)
    # Fill this mapping when you download Nexar (column names differ by release).
    colmap = {
        "clip_id": "clip_id",
        "path": "path",
        "label": "label",
    }
    out = pd.DataFrame()
    for dest in CANON:
        src = colmap.get(dest, dest)
        out[dest] = raw[src] if src in raw.columns else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()
