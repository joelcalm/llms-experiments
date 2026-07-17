# HTCondor batch path

Submit files and worker scripts to run the `experiment_cli.py` matrix on an
HTCondor GPU cluster, one job per model.

These are **deployment templates**: paths resolve from a single `RUN_ROOT`
environment variable rather than hardcoded cluster paths, so nothing here is
tied to a specific site or account. Export it before submitting:

```bash
export RUN_ROOT=/path/to/your/run_root   # holds code/, results/, logs/, status/
condor_submit condor/submit_full_matrix_h100.sub
```

`RUN_ROOT` is expanded at submit time (HTCondor `$ENV(RUN_ROOT)`) into
`initialdir`, `executable`, and the log paths, and passed into each job's
environment. The expected layout under `RUN_ROOT`:

```text
$RUN_ROOT/
  code/                     # this repository, checked out on shared storage
    .venv/                  # environment built once (see submit_when_ready.sh)
    condor/                 # these scripts
    experiment-cli/, experiments/
  results/  logs/  status/  # created by the workers
```

## Files

- `submit_*.sub` — one submit file per hardware/model batch; each `queue`s the
  models to run. Edit the `queue … from (…)` block to choose models.
- `run_matrix_worker.sh` — per-slot entry point a `.sub` executes; downloads the
  model and runs `experiment_cli.py run-matrix` for it.
- `submit_when_ready.sh` — waits for the one-time environment build to finish,
  then submits the matrix.

## Caches (optional)

Model weights and compile artefacts default to `$RUN_ROOT/shared_cache` and
`$RUN_ROOT/hf_cache`. To reuse a persistent cache shared across runs, export
`SHARED_CACHE_ROOT` and/or `HF_HOME` before submitting.
