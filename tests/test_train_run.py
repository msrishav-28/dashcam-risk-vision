import json
from pathlib import Path

import pytest

from dashcam_risk.train import main

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "processed" / "example_manifest.csv"

_FAST = [
    "--manifest", str(MANIFEST),
    "--model", "majority",
    "--epochs", "1",
    "--seeds", "42,43",
]


def test_placeholder_run_is_stamped_and_provenance_rich(scratch_dir):
    out = main([*_FAST, "--allow-placeholder", "--experiments-dir", str(scratch_dir)])
    assert out.exists()
    assert "PLACEHOLDER" in out.parent.name, "run folder must carry the stamp"

    rec = json.loads(out.read_text())
    assert rec["placeholder_data"] is True
    assert "not a result" in rec["claim_note"]
    assert rec["command"]
    assert rec["started_utc"] and rec["ended_utc"]
    assert rec["manifest"]["sha256"]
    assert rec["manifest"]["clip_ids"] == 10
    assert rec["splits"]["hash"]
    assert rec["splits"]["counts"]["test"]["clips"] > 0
    assert rec["git"]["sha"] is not None or rec["git"]["note"] == "not a git repository"
    assert rec["device"]["cpu_count"] and rec["libraries"]["torch"]

    assert len(rec["per_seed"]) == 2
    for row in rec["per_seed"]:
        assert row["split_hash"] and row["metrics"]["threshold"] is not None
    agg = rec["test_metrics_aggregate"]
    assert agg["n_seeds"] == 2
    for key in ("test_auprc", "test_recall_lead_ge_1.0s",
                "test_recall_lead_ge_1.5s", "test_recall_lead_ge_2.0s"):
        assert key in agg["mean"], f"missing {key}"
    # the headline "how early" metrics are recorded — the point of the study
    assert "tte_mae_constant_baseline_s" in agg["mean"]


def test_runs_never_overwrite(scratch_dir):
    argv = [*_FAST, "--allow-placeholder", "--experiments-dir", str(scratch_dir)]
    first = main(argv)
    second = main(argv)
    assert first != second
    assert first.exists() and second.exists()


def test_missing_tensors_fail_loudly_without_opt_in(scratch_dir):
    with pytest.raises(FileNotFoundError, match="allow_placeholder"):
        main([*_FAST, "--experiments-dir", str(scratch_dir)])


def test_night_holdout_mode_flags_small_slice(scratch_dir):
    out = main([
        *_FAST,
        "--allow-placeholder",
        "--holdout-column", "time_of_day",
        "--holdout-values", "night",
        "--experiments-dir", str(scratch_dir),
    ])
    rec = json.loads(out.read_text())
    assert rec["splits"]["mode"] == "holdout"
    assert rec["splits"]["holdout"]["held_out_values"] == ["night"]
    assert rec["splits"]["holdout"]["test_slice_counts"].get("night") == 5
    assert rec["slice_warning"], "5-clip night slice must be flagged unreliable"
