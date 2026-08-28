#!/usr/bin/env bash
# Example: run a single configuration on one local GPU.
#
# This is a tracked template, not a generated job. Adapt CONFIG/OUTPUT_DIR and
# submit it on any machine with a CUDA GPU and the `gpu` extra installed. It
# calls the same `llms-experiments run` entry point used by the HTCondor and
# Slurm wrappers, so outputs obey the same Parquet result contract.
#
# Usage:
#   export CONFIG=configs/protoethos/protoethos_full_matrix.yaml
#   export OUTPUT_DIR=<durable-results>
#   ./examples/run_local_gpu.sh
#
# For a quick smoke test with the deterministic offline backend (no GPU, no
# model download), use:
#   CONFIG=configs/protoethos/protoethos_full_matrix.yaml \
#   OUTPUT_DIR=/tmp/smoke ./examples/run_local_gpu.sh --backend fake --rows 4
set -euo pipefail

: "${CONFIG:?Set CONFIG to a run YAML file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"

# Defaults; override via environment.
GPUS="${GPUS:-1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-96G}"
BACKEND="${BACKEND:-vllm}"
ROWS_ARG=()
if [[ -n "${ROWS:-}" ]]; then
  ROWS_ARG=(--rows "$ROWS")
fi

export OMP_NUM_THREADS="${CPUS}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

args=(
  llms-experiments run "$CONFIG"
  --backend "$BACKEND"
  --output "$OUTPUT_DIR"
  "${ROWS_ARG[@]}"
)

# `uv run` honours the project environment and the `gpu` extra when installed.
exec uv run "${args[@]}"
