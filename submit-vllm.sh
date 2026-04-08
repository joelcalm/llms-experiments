#!/bin/bash
#SBATCH --job-name=vllm-scores
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch submit-vllm-cesga.sh [PROMPT_TYPE] [INPUT_CSV] [OUTPUT_CSV] [MODEL] [extra batch_infer_simple.py args...]
# Example:
#   sbatch submit-vllm-cesga.sh MFT v2_3m_final_clean_text.csv outputs/mft_scores.csv qwen-2b-awq --limit 50000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROMPT_TYPE="${1:-MFT}"
INPUT_CSV="${2:-v2_3m_final_clean_text.csv}"
OUTPUT_CSV="${3:-outputs/${PROMPT_TYPE,,}_scores_${SLURM_JOB_ID:-local}.csv}"
MODEL="${4:-qwen-2b-awq}"

if [[ $# -ge 4 ]]; then
    shift 4
else
    shift $#
fi
EXTRA_ARGS=("$@")

mkdir -p logs outputs

# CESGA typically expects software loaded via spack. Keep these optional so the script
# still works if your environment is already activated elsewhere.
if command -v spack >/dev/null 2>&1; then
    spack load python@3.11 || true
    spack load cuda@12.4 || true
fi

if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-n/a}"
echo "Prompt type: $PROMPT_TYPE"
echo "Input CSV:   $INPUT_CSV"
echo "Output CSV:  $OUTPUT_CSV"
echo "Model:       $MODEL"

srun python3 batch_infer_simple.py \
    --input-csv "$INPUT_CSV" \
    --text-column text \
    --output-csv "$OUTPUT_CSV" \
    --prompt-md prompt_examples_og.md \
    --prompt-type "$PROMPT_TYPE" \
    --model "$MODEL" \
    --runtime-profile a100 \
    --gpu-mem-util 0.94 \
    --max-model-len 1024 \
    --max-num-seqs 64 \
    --batch-size 128 \
    --max-tokens 96 \
    --context-reserve-tokens 128 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --resume \
    --truncate-to-fit \
    --flush-every 8 \
    --log-every 5000 \
    "${EXTRA_ARGS[@]}"
