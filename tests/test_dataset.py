import pandas as pd
import pytest
import torch

from dashcam_risk.dataset import ClipFrameDataset


def _df(rows):
    return pd.DataFrame(rows)


def _risk_row(clip_id="c000", alert=12.0, event=18.0):
    return {
        "clip_id": clip_id,
        "path": f"placeholder/{clip_id}.pt",
        "label": "risk",
        "duration_s": 40.0,
        "event_time_s": event,
        "alert_time_s": alert,
    }


def test_missing_tensor_fails_loudly_by_default():
    ds = ClipFrameDataset(_df([_risk_row()]))
    with pytest.raises(FileNotFoundError, match="allow_placeholder"):
        ds[0]


def test_placeholder_is_opt_in_and_repeatable():
    df = _df([_risk_row("c000"), _risk_row("c001")])
    a = ClipFrameDataset(df, allow_placeholder=True)
    b = ClipFrameDataset(df, allow_placeholder=True)
    ia, ib = a[0], b[0]
    assert ia["frames"].shape == (8, 3, 224, 224)
    assert torch.equal(ia["frames"], ib["frames"]), "placeholder must be deterministic"
    assert a.used_placeholder


def test_placeholder_counter_and_label():
    df = _df([_risk_row("c000")])
    ds = ClipFrameDataset(df, allow_placeholder=True)
    item = ds[0]
    assert ds.n_placeholder == 1
    assert ds.used_placeholder
    assert item["y"].item() == 1
    assert item["t_last"].item() < 12.0
    assert item["lead_s"].item() == pytest.approx(18.0 - item["t_last"].item())


def test_real_tensor_file_loads_without_placeholder(scratch_dir):
    p = scratch_dir / "c000.pt"
    torch.save(torch.zeros(8, 3, 224, 224), p)
    row = _risk_row("c000")
    row["path"] = str(p)
    ds = ClipFrameDataset(_df([row]), allow_placeholder=True)
    item = ds[0]
    assert item["frames"].abs().sum().item() == 0.0
    assert ds.n_placeholder == 0
    assert not ds.used_placeholder


def test_t_last_respects_alert_on_any_label():
    row = {
        "clip_id": "n0",
        "path": "placeholder/n0.pt",
        "label": "normal",
        "duration_s": 40.0,
        "alert_time_s": 10.0,
    }
    ds = ClipFrameDataset(_df([row]), n_frames=8, allow_placeholder=True)
    assert ds[0]["t_last"].item() < 10.0
