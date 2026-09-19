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
python scripts/train_baseline.py --manifest data/processed/example_manifest.csv --epochs 1 --model gru
```

The example manifest has no videos. The dataset class fills **placeholder tensors**
so the trainer and tests run before you download Nexar (or another licensed set).
Do not quote that run as a result.

## Layout

```
docs/           QUESTION, TASK, LEAKAGE, DATA_CARD, NOT_CLAIMED
configs/        3050-scale defaults
src/dashcam_risk/
  schema.py     clip record + cutoff
  leakage.py    invariants
  splits.py     clip-level and slice holdouts
  metrics.py    AUPRC, lead-time recall, TTE MAE
  models.py     last-frame and frame-GRU stubs
  dataset.py    T-frame loader
  train.py      tiny loop that writes experiments/last_run.json
scripts/
tests/
data/processed/example_manifest.csv
```

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
