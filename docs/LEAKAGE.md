# Leakage rules

These are code invariants (`src/dashcam_risk/leakage.py` + tests). If a test
fails, the run is invalid even if AUPRC looks good.

## Forbidden

1. Putting frames at or after `alert_time_s` into the model input on risk clips.
2. Splitting by *frame* or *second* so the same `clip_id` appears in train and test.
3. Using `event_time_s`, outcome text, or post-event telemetry as a feature.
4. Sampling more frames from risk clips than from normal clips as a hidden cue.
5. Sorting or padding in a way that sequence length equals the label.

## Required

- Split key is `clip_id` (and `trip_id` if present).
- Train / val / test clip ID sets are disjoint.
- Every training example records `t_last_input_s < alert_time_s` when alert exists.
- Frame count and image size are identical across classes unless documented.

## Why this is here

A random 80/20 frame split on dashcam video is how pothole-YOLO repos fake mAP.
The interesting failure is a model that looks at the crash itself.
