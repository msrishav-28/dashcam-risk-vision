# Data (not committed)

Keep raw video off git.

```
data/
  raw/           # dataset downloads (Nexar or local clips)
  interim/       # extracted frames, per-clip folders
  processed/     # manifests + splits (small CSVs may be committed later)
```

Expected processed manifest columns are listed in `docs/DATA_CARD.md`.

Do not commit mp4/jpg from dashcams that include identifiable plates or faces
unless you have a documented licence and a redaction step.
