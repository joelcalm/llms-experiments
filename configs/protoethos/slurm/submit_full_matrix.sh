#!/usr/bin/env bash
#SBATCH --job-name=protoethos-matrix
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${CONFIG:=configs/protoethos/ministral_all_datasets.yaml}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"

args=(llms-experiments run "$CONFIG" --output "$OUTPUT_DIR")
if [[ -n "${MODEL_ID:-}" ]]; then
    args+=(--model "$MODEL_ID")
fi
if [[ -n "${DATASETS:-}" ]]; then
    args+=(--datasets "$DATASETS")
fi
if [[ -n "${VARIANTS:-}" ]]; then
    args+=(--variants "$VARIANTS")
fi

export HF_HOME="${HF_HOME:-${TMPDIR:-/tmp}/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

srun "${args[@]}"
