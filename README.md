# dashcam-risk-vision

Clip-level dashcam **collision / near-miss risk scoring** with leakage-aware splits.

This is a starter for a computer-vision + automobile study. It is **not** ADAS,
not a CARLA Autopilot, not occupancy prediction, and not a pothole YOLO.

Until `experiments/` contains a run on real clips, there are **no performance claims**.

## Question

Given only frames *before* the dataset's alert time, can a small video model
score whether a clip will become a collision or near-miss — and how early?

See [`docs/QUESTION.md`](docs/QUESTION.md) and [`docs/TASK.md`](docs/TASK.md).

## Why the protocol exists

Random frame splits let the model watch the crash. That is leakage.
`docs/LEAKAGE.md` and `tests/test_leakage.py` encode the rules:

- split by `clip_id`, never by frame
- last input timestamp must be `< alert_time_s` on risk clips
- sequence length is fixed so duration cannot be the label

## Setup

Python 3.10+. A 4–6 GB GPU is enough for the stub (`T=8`, `224²`).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/check_leakage.py --manifest data/processed/example_manifest.csv
python scripts/train_baseline.py --manifest data/processed/example_manifest.csv --epochs 1 --model gru --allow-placeholder --seeds 42
```

The example manifest has no videos. Missing frame tensors make the trainer
**fail loudly**; `--allow-placeholder` opts in to deterministic stand-in tensors
so the machinery runs before you download Nexar (or another licensed set).
Placeholder runs are stamped `placeholder_data: true` in the run record and
folder name. Do not quote them as results.

## Layout

```
docs/           QUESTION, TASK, LEAKAGE, DATA_CARD, NOT_CLAIMED
src/dashcam_risk/
  schema.py     clip record + cutoff
  leakage.py    invariants (alert-time guard applies to any clip carrying alert_time_s)
  splits.py     clip-level splits, slice holdouts, small-slice flag
  metrics.py    AUPRC, lead-time recall, TTE MAE, threshold pick, seed aggregation
  models.py     majority / last-frame / frame-GRU baselines
  dataset.py    T-frame loader; placeholder tensors only with --allow-placeholder
  train.py      seeded loop writing immutable run records under experiments/runs/
scripts/
tests/
data/processed/example_manifest.csv
```

## Run records

Every train writes a new immutable folder `experiments/runs/<UTC>_<model>_<PLACEHOLDER|REAL>/run.json`
— never overwriting earlier runs. The record carries the exact command, git SHA,
manifest hash, split hash, seeds, per-seed metrics with mean/std across seeds,
library versions, hardware, and timestamps. Quote numbers only from these files,
with the `placeholder_data` field visible. See `experiments/README.md`.

Domain-shift exam: add `--holdout-column time_of_day --holdout-values night` to
evaluate on a held-out slice; slices under 30 clips are flagged unreliable.

## Data

Intended public source: [Nexar Dashcam Collision Prediction](https://arxiv.org/abs/2503.03848).
Map their columns in `scripts/build_manifest.py`. Keep mp4s out of git.
Schema: [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Baselines to log before any transformer

1. Majority class
2. `--model last_frame`
3. `--model gru`
4. Then night / rain holdout via `holdout_by_column`

Report AUPRC (imbalance) and lead-time recall at 1 s / 1.5 s / 2 s — not accuracy.

## Bound claims

[`docs/NOT_CLAIMED.md`](docs/NOT_CLAIMED.md). This repo scores public dashcam clips.
It does not brake a car.

## License

MIT for the code. Dataset licences are separate.
