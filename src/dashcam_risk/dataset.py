from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dashcam_risk.leakage import assert_input_before_alert, sample_times
from dashcam_risk.schema import ClipRecord


def row_to_record(row: pd.Series) -> ClipRecord:
    def f(name):
        val = row[name] if name in row.index else None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return val

    def fnum(name):
        val = f(name)
        return None if val is None else float(val)

    return ClipRecord(
        clip_id=str(row["clip_id"]),
        path=str(row["path"]),
        label=str(row["label"]),
        trip_id=None if f("trip_id") is None else str(f("trip_id")),
        event_type=None if f("event_type") is None else str(f("event_type")),
        weather=None if f("weather") is None else str(f("weather")),
        time_of_day=None if f("time_of_day") is None else str(f("time_of_day")),
        scene=None if f("scene") is None else str(f("scene")),
        duration_s=fnum("duration_s"),
        fps=fnum("fps"),
        event_time_s=fnum("event_time_s"),
        alert_time_s=fnum("alert_time_s"),
        license=None if f("license") is None else str(f("license")),
    )


class ClipFrameDataset(Dataset):
    """Loads a precomputed (T,C,H,W) tensor per clip, or a placeholder."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        n_frames: int = 8,
        image_size: int = 224,
        root: str | Path = ".",
    ):
        self.df = manifest.reset_index(drop=True)
        self.n_frames = n_frames
        self.image_size = image_size
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        rec = row_to_record(self.df.iloc[idx])
        duration = rec.duration_s if rec.duration_s is not None else 40.0
        times = sample_times(duration, rec.cutoff_s(), self.n_frames)
        if rec.is_risk:
            assert_input_before_alert(times[-1], rec)
        tensor_path = self.root / rec.path
        if tensor_path.suffix == ".pt" and tensor_path.exists():
            frames = torch.load(tensor_path, map_location="cpu")
        else:
            # Deterministic placeholder so the trainer runs before you extract frames.
            g = torch.Generator().manual_seed(abs(hash(rec.clip_id)) % (2**31))
            frames = torch.rand(self.n_frames, 3, self.image_size, self.image_size, generator=g)
        if frames.ndim != 4:
            raise ValueError(f"expected TCHW tensor at {tensor_path}")
        y = 1 if rec.is_risk else 0
        lead = None
        if rec.event_time_s is not None:
            lead = max(rec.event_time_s - times[-1], 0.0)
        return {
            "frames": frames.float(),
            "y": torch.tensor(y, dtype=torch.long),
            "clip_id": rec.clip_id,
            "t_last": torch.tensor(times[-1], dtype=torch.float32),
            "lead_s": torch.tensor(lead if lead is not None else -1.0),
        }
