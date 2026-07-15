# LLMs Experiments

A configuration-driven runner and script utilities for local Large Language Model experiments.

## Project Structure
- `experiment-cli/`: Configuration-driven CLI tool for running and validation of local LLM models.
- `slurm/`: Slurm cluster batch-inference path — `submit-vllm.sh` (job submission) with its companion
  `batch_infer_simple.py`, `sample_csv_deterministic.py`, and one-label prompt templates under `slurm/prompts/`.
- `scripts/legacy/`: Older standalone single-sentence prototypes (`annotate.py`, `run_vllm.py`), kept for reference.
- `models/`: Model list configurations.
- `prompts_legacy/`: Archived demo prompt examples used by the legacy prototypes.
- `docs/`: Auxiliary documentation including codebase architecture and classification strategies.


## Setup and Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and local virtual environment isolation.

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Initialize Local Virtual Environment**:
   ```bash
   uv venv
   ```

3. **Install Dependencies**:
   ```bash
   UV_CACHE_DIR=.uv-cache uv sync
   UV_CACHE_DIR=.uv-cache uv run python experiment-cli/experiment_cli.py gpu-preflight
   ```

   The installation is not GPU-ready until `gpu-preflight` succeeds. It checks
   both `nvidia-smi` (the host driver and `/dev/nvidia*` device nodes) and the
   CUDA-enabled PyTorch package installed by uv. uv manages Python and CUDA
   user-space libraries; it cannot install the kernel-level NVIDIA driver.
   Repair the host driver if the preflight fails, then rerun it before launching
   an experiment.

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

## Generic experiment CLI

The self-contained local LLM runner lives in [`experiment-cli/`](experiment-cli/). Its [usage guide](experiment-cli/README.md) includes a uv quick start, a no-GPU self-test, the full local-GPU command, and the output contract.
