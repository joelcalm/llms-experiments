# LLMs Experiments

A configuration-driven runner and script utilities for local Large Language Model experiments.
One YAML file chooses the input, model, prompts, output schemas, and batching; the runner
([`experiment-cli/`](experiment-cli/)) stays domain-agnostic.

## Getting Started

A short path from zero to a verified run. **Steps 1–3 need no GPU** — they prove the install
and the full pipeline on a built-in fake backend.

1. **Install [uv](https://github.com/astral-sh/uv)** (skip if you already have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies.** `uv sync` creates `.venv/` and installs everything:
   ```bash
   uv sync                    # GPU-capable environment (default)
   uv sync --no-group gpu     # CPU-only: 28 packages instead of 191, skips ~9 GB of CUDA wheels
   ```
   The CPU-only install is enough for `validate`, `prepare`, `parse`, the `nvidia_api`
   backend, and the whole test suite.

3. **Verify without a GPU.** The self-test runs all five request modes and resume logic on the
   fake backend and writes only temporary files:
   ```bash
   uv run python experiment-cli/experiment_cli.py self-test \
     --config experiments/local_all_modes_smoke.yaml
   ```
   A `Self-test passed` line means the install is good.

4. **Confirm the GPU** (only for local inference). This gates a real run:
   ```bash
   uv run python experiment-cli/experiment_cli.py gpu-preflight
   ```
   It checks `nvidia-smi` (host driver and `/dev/nvidia*`) and CUDA access from uv's PyTorch.
   uv installs CUDA user-space libraries but not the kernel driver; if it fails, repair the
   host driver and rerun before launching an experiment.

5. **Run the smoke experiment** on the GPU:
   ```bash
   uv run python experiment-cli/experiment_cli.py run \
     --config experiments/local_all_modes_smoke.yaml
   ```

Every command follows the same shape — `experiment_cli.py <command> --config <file>` — and the
same overrides (`--set key=value`, `--run-id`, `--output`, `--backend`). The
[experiment-cli guide](experiment-cli/README.md) covers configs, the batch/API paths, the
five-dataset matrix, and the output contract.

> uv caches wheels into a gitignored `.uv-cache/` inside the project (set by `[tool.uv]
> cache-dir`), so no command needs a `UV_CACHE_DIR=` prefix.

## Project Structure
- `experiment-cli/`: Configuration-driven CLI tool for running and validation of local LLM models.
- `condor/`: HTCondor submit files (`.sub`) and worker scripts for an HTCondor GPU cluster.
  Each `.sub` queues one `experiment_cli.py run-matrix` worker per model; `run_matrix_worker.sh` is the
  per-slot entry point. Paths resolve from `RUN_ROOT`, so the files carry no site-specific details — see
  [`condor/README.md`](condor/README.md). Submit with `condor_submit condor/<file>.sub`.
- `slurm/`: Slurm cluster batch-inference path — `submit-vllm.sh` (job submission) with its companion
  `batch_infer_simple.py`, `sample_csv_deterministic.py`, and one-label prompt templates under `slurm/prompts/`.
  Submit with `sbatch slurm/submit-vllm.sh`. The two schedulers are kept in separate folders so each
  cluster's jobs can be submitted without cross-contamination.
- `scripts/legacy/`: Older standalone single-sentence prototypes (`annotate.py`, `run_vllm.py`), kept for reference.
- `models/`: Model list configurations.
- `prompts_legacy/`: Archived demo prompt examples used by the legacy prototypes.
- `docs/`: Auxiliary documentation including codebase architecture and classification strategies.


## Development and Verification

- **Linting**: Run Ruff check to enforce code style:
  ```bash
  uv run ruff check .
  ```

- **Formatting**: Run Ruff format check:
  ```bash
  uv run ruff format --check .
  ```

- **Testing**: Run pytest:
  ```bash
  uv run pytest
  ```

  The suite runs entirely on the `fake` backend, so it needs no GPU. It includes
  golden snapshots (`tests/golden/`) that pin the runner's observable output —
  result rows and manifests — for the smoke config and for all five dataset
  lanes of the matrix config. If a change to the runner is *meant* to alter that
  output, review the diff and re-record it:

  ```bash
  uv run pytest --golden-update
  ```

  On a machine that sources ROS (`/opt/ros/*/setup.bash`), `PYTHONPATH` leaks
  ROS's Python 3.12 site-packages into this 3.13 environment and pytest fails
  while auto-loading ROS's plugins. Clear it for the command:

  ```bash
  PYTHONPATH= uv run pytest
  ```
