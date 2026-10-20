One folder per run: `runs/<UTC timestamp>_<model>_<PLACEHOLDER|REAL>/run.json`.

- `PLACEHOLDER` runs are smoke tests on dummy tensors — never results.
- `REAL` runs used actual frame tensors; still check `manifest.license_values`
  before quoting anything.
- `run.json` carries: command, git SHA (+ dirty flag), manifest sha256, split
  hash, seeds, per-seed metrics, mean/std across seeds, threshold (chosen on
  validation only), library versions, hardware, UTC start/end.
- Never edit or delete a run folder; new runs get new folders.
- Report numbers only from these files, with the `placeholder_data` field visible.
