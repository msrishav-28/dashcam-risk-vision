# Tasks

## A — Clip classification

Binary label: `risk` (collision or near-miss) vs `normal`.

Input is a sampled frame sequence **strictly before** `alert_time_s`.
If `alert_time_s` is missing on normal clips, use the full clip but still sample
a fixed window from the start so length does not leak the label.

## B — Time-to-event

On risk clips, regress seconds from the last input frame to `event_time_s`.
Report MAE and whether the alert would have been ≥1.5 s and ≥2.0 s early.

## C — Domain shift

Hold out at least one slice the model is not tuned on:

- night vs day
- rain vs clear
- highway vs urban
- optional later: a non-Nexar / India dashcam set

The **drop** is the result. Do not hide it.

## D — Explainability (after A works)

Grad-CAM or attention over the sampled frames. Save grids for true positives
and for late alerts (score rises only after a human would already have seen it).

## Baselines (run these first)

1. Majority class
2. Last-frame-only ImageNet classifier (`vit_tiny` / EfficientNet-B0)
3. Three-frame CNN + GRU

A transformer is allowed only after (2) and (3) are in `experiments/`.
