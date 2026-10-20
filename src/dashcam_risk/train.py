from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dashcam_risk.dataset import ClipFrameDataset
from dashcam_risk.metrics import (
    aggregate_seed_runs,
    choose_threshold_f1,
    classification_report,
    lead_time_recall,
    time_to_event_mae,
)
from dashcam_risk.models import build_model
from dashcam_risk.schema import RISK_LABELS
from dashcam_risk.splits import (
    clip_level_split,
    holdout_by_column,
    load_manifest,
    slice_warning,
    split_counts,
    train_val_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETERMINISM_NOTE = (
    "python/numpy/torch seeded per run; cudnn.deterministic=True; "
    "DataLoader shuffle uses a seeded generator; some CUDA ops are "
    "nondeterministic regardless of seeding"
)

LEAD_KEYS = tuple(f"recall_lead_ge_{h:.1f}s" for h in (1.0, 1.5, 2.0))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, risk_prob, lead_s) for every item; lead_s is -1 on non-risk."""
    model.eval()
    ys, ps, leads = [], [], []
    for batch in loader:
        logits = model(batch["frames"].to(device))
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(prob)
        leads.append(batch["lead_s"].numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    lead = np.concatenate(leads) if leads else np.array([])
    return y, p, lead


def metrics_at(y: np.ndarray, p: np.ndarray, lead: np.ndarray, threshold: float) -> dict:
    rep = classification_report(y, p) if len(y) else {"n": 0.0}
    risk = (y == 1) & (lead >= 0)
    if len(y) and risk.any():
        rep.update(lead_time_recall(y == 1, np.where(risk, lead, -1.0), p, threshold))
    else:
        for k in LEAD_KEYS:
            rep[k] = float("nan")
    return rep


def constant_tte_baseline(train_y, train_lead, test_y, test_lead) -> float:
    """Task-B floor: predict every test clip's time-to-event as the mean train lead.

    A real time-to-event model must beat this. NaN when either side has no
    risk clips with a known event time.
    """
    tr = train_lead[(train_y == 1) & (train_lead >= 0)]
    te = test_lead[(test_y == 1) & (test_lead >= 0)]
    if len(tr) == 0 or len(te) == 0:
        return float("nan")
    return time_to_event_mae(te, np.full_like(te, tr.mean()))


def build_splits(df: pd.DataFrame, args, seed: int) -> dict[str, pd.DataFrame]:
    if args.holdout_column:
        if not args.holdout_values:
            raise ValueError("--holdout-values is required when --holdout-column is set")
        trainval, test = holdout_by_column(df, args.holdout_column, args.holdout_values)
        train, val = train_val_split(trainval, seed=seed, val_size=args.val_size)
        parts = {"train": train, "val": val, "test": test}
        for name, part in parts.items():
            part["split"] = name
        return parts
    return clip_level_split(df, seed=seed, val_size=args.val_size, test_size=args.test_size)


def run_single_seed(seed: int, df: pd.DataFrame, args, device) -> dict:
    set_seeds(seed)
    parts = build_splits(df, args, seed)

    def make_ds(part):
        return ClipFrameDataset(
            part,
            n_frames=args.n_frames,
            image_size=args.image_size,
            root=args.root,
            allow_placeholder=args.allow_placeholder,
        )

    train_ds = make_ds(parts["train"])
    val_ds = make_ds(parts["val"])
    test_ds = make_ds(parts["test"])

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args.model).to(device)
    fit_prior = getattr(model, "fit_prior", None)
    if callable(fit_prior):
        train_labels = (
            parts["train"].drop_duplicates("clip_id")["label"].astype(str).str.lower().isin(RISK_LABELS)
        )
        fit_prior(torch.from_numpy(train_labels.to_numpy(dtype=np.int64)))

    # MajorityClassClassifier has no trainable parameters by design; newer
    # torch raises on optimizers built from an empty parameter list.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr) if trainable else None
    loss_fn = torch.nn.CrossEntropyLoss()
    has_trainable = opt is not None

    history = []
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            logits = model(batch["frames"].to(device))
            loss = loss_fn(logits, batch["y"].to(device))
            if has_trainable:
                opt.zero_grad()
                loss.backward()
                opt.step()
            running += float(loss.item()) * len(batch["y"])
            n += len(batch["y"])
        vy, vp, _ = predict(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(n, 1),
                "val_auprc": classification_report(vy, vp)["auprc"],
            }
        )

    train_y, _, train_lead = predict(model, train_loader, device)
    val_y, val_p, val_lead = predict(model, val_loader, device)
    threshold = choose_threshold_f1(val_y, val_p)
    test_y, test_p, test_lead = predict(model, test_loader, device)

    val_rep = metrics_at(val_y, val_p, val_lead, threshold)
    test_rep = metrics_at(test_y, test_p, test_lead, threshold)
    tte = constant_tte_baseline(train_y, train_lead, test_y, test_lead)

    flat = {f"val_{k}": v for k, v in val_rep.items()}
    flat.update({f"test_{k}": v for k, v in test_rep.items()})
    flat["threshold"] = threshold
    flat["tte_mae_constant_baseline_s"] = tte

    return {
        "seed": seed,
        "split_hash": split_hash(parts),
        "metrics": flat,
        "history": history,
        "placeholder_used": train_ds.used_placeholder or val_ds.used_placeholder or test_ds.used_placeholder,
        "n_placeholder_tensors": train_ds.n_placeholder + val_ds.n_placeholder + test_ds.n_placeholder,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    sha = run("git", "rev-parse", "HEAD")
    if sha.returncode != 0:
        return {"sha": None, "dirty": None, "note": "not a git repository"}
    status = run("git", "status", "--porcelain")
    return {"sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}


def split_hash(parts: dict[str, pd.DataFrame]) -> str:
    h = hashlib.sha256()
    for name in sorted(parts):
        ids = sorted(set(parts[name]["clip_id"].astype(str)))
        h.update(f"{name}:{','.join(ids)}\n".encode("utf-8"))
    return h.hexdigest()


def library_versions() -> dict:
    out = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "timm"):
        try:
            out[name] = pkg_version(name)
        except Exception as e:  # package absent: record why, never guess
            out[name] = f"unavailable ({type(e).__name__})"
    return out


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(s) for s in raw.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"--seeds must be comma-separated integers, got {raw!r}")
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def main(argv: list[str] | None = None) -> Path:
    started = datetime.now(timezone.utc)
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/example_manifest.csv")
    p.add_argument("--model", default="gru", choices=["majority", "last_frame", "gru"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--root", default=".", help="data root that manifest paths are relative to")
    p.add_argument("--allow-placeholder", action="store_true",
                   help="opt-in: fill missing tensors with deterministic stand-ins. "
                        "Runs are stamped placeholder_data=true and are never results.")
    p.add_argument("--holdout-column", default=None,
                   help="e.g. time_of_day — evaluate on this slice only (domain-shift exam)")
    p.add_argument("--holdout-values", nargs="*", default=None,
                   help="values of --holdout-column held out, e.g. night rain")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--experiments-dir", default="experiments")
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    df = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = []
    for seed in seeds:
        per_seed.append(run_single_seed(seed, df, args, device))
        print(f"seed {seed} done")

    placeholder_used = any(s["placeholder_used"] for s in per_seed)
    n_placeholder = sum(s["n_placeholder_tensors"] for s in per_seed)
    aggregate = aggregate_seed_runs([s["metrics"] for s in per_seed])

    parts_for_record = build_splits(df, args, seeds[0])
    holdout_info = None
    if args.holdout_column:
        holdout_info = {
            "column": args.holdout_column,
            "held_out_values": list(args.holdout_values or []),
            "test_slice_counts": parts_for_record["test"][args.holdout_column]
            .astype(str).value_counts().to_dict(),
        }
    warn = None
    if args.holdout_column:
        warn = slice_warning(int(parts_for_record["test"]["clip_id"].nunique()))

    manifest_licenses = (
        sorted(df["license"].dropna().astype(str).unique().tolist())
        if "license" in df.columns else []
    )

    record = {
        "schema_version": 1,
        "placeholder_data": placeholder_used,
        "n_placeholder_tensors": n_placeholder,
        "claim_note": (
            "SMOKE TEST on placeholder tensors - not a result"
            if placeholder_used
            else "run on real frame tensors"
        ),
        "command": " ".join(shlex.quote(a) for a in sys.argv if a != "-m"),
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items()},
        "git": git_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "device": {
            "name": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
        },
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
            "rows": int(len(df)),
            "clip_ids": int(df["clip_id"].nunique()),
            "license_values": manifest_licenses,
        },
        "splits": {
            "mode": "holdout" if args.holdout_column else "random",
            "counts": split_counts(parts_for_record),
            "hash": split_hash(parts_for_record),
            "holdout": holdout_info,
        },
        "slice_warning": warn,
        "seeds": seeds,
        "per_seed": [
            {
                "seed": s["seed"],
                "split_hash": s["split_hash"],
                "metrics": s["metrics"],
                "history": s["history"],
            }
            for s in per_seed
        ],
        "test_metrics_aggregate": aggregate,
        "determinism_note": DETERMINISM_NOTE,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "PLACEHOLDER" if placeholder_used else "REAL"
    base = Path(args.experiments_dir) / "runs" / f"{stamp}_{args.model}_{tag}"
    run_dir, i = base, 2
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}-{i}")
        i += 1
    run_dir.mkdir(parents=True)
    out = run_dir / "run.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")

    def _fmt(v: float) -> str:
        return f"{v:.3f}" if v == v else "n/a"  # v != v detects NaN (single seed)

    agg = aggregate["mean"]
    print(
        "test AUPRC {}+/-{} | lead>=1.5s {}+/-{} | {}".format(
            _fmt(agg.get("test_auprc", float("nan"))),
            _fmt(aggregate["std"].get("test_auprc", float("nan"))),
            _fmt(agg.get("test_recall_lead_ge_1.5s", float("nan"))),
            _fmt(aggregate["std"].get("test_recall_lead_ge_1.5s", float("nan"))),
            "PLACEHOLDER - not a result" if placeholder_used else "REAL data",
        )
    )
    return out


if __name__ == "__main__":
    main()
