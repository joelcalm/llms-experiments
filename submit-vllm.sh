#!/bin/bash
#SBATCH --job-name=vllm-scores
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch submit-vllm.sh [PROMPT_TYPE] [INPUT_CSV] [OUTPUT_CSV] [MODEL] [extra batch_infer_simple.py args...]
# Example:
#   sbatch submit-vllm.sh MFT v2_3m_final_clean_text.csv outputs/mft_scores.csv qwen-2b-awq --limit 50000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SLURM_SUBMIT_DIR:-$ROOT_DIR}"
cd "$RUN_DIR"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Run dir does not exist: $RUN_DIR" >&2
    exit 2
fi

if [[ ! -f "$RUN_DIR/batch_infer_simple.py" ]]; then
    echo "ERROR: Missing $RUN_DIR/batch_infer_simple.py" >&2
    echo "Hint: submit from the project root so SLURM_SUBMIT_DIR points at llms-experiments." >&2
    exit 2
fi

if [[ ! -f "$RUN_DIR/prompt_examples.md" ]]; then
    echo "ERROR: Missing $RUN_DIR/prompt_examples.md" >&2
    exit 2
fi

PROMPT_TYPE="${1:-MFT}"
INPUT_CSV="${2:-v2_3m_final_clean_text.csv}"
OUTPUT_CSV="${3:-$RUN_DIR/outputs/${PROMPT_TYPE,,}_scores_${SLURM_JOB_ID:-local}.csv}"
MODEL="${4:-qwen-2b-awq}"

if [[ $# -ge 4 ]]; then
    shift 4
else
    shift $#
fi
EXTRA_ARGS=("$@")

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs"

# CESGA typically expects software loaded via spack. Keep these optional so the script
# still works if your environment is already activated elsewhere.
if command -v spack >/dev/null 2>&1; then
    spack load python@3.11 || true
    spack load cuda@12.4 || true
fi

ACTIVATED_ENV="myenv"
if [[ -f "$ACTIVATED_ENV/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$ACTIVATED_ENV/bin/activate"
    if ! python3 -c "import vllm" >/dev/null 2>&1; then
        deactivate >/dev/null 2>&1 || true
        echo "ERROR: 'myenv' was found but cannot import 'vllm'." >&2
        echo "Reinstall in myenv: module load cesga/2022 python/3.10.8 && python3 -m venv myenv && source myenv/bin/activate && python3 -m pip install --no-cache-dir vllm" >&2
        exit 1
    fi
else
    echo "ERROR: Missing myenv/bin/activate" >&2
    echo "Create myenv: module load cesga/2022 python/3.10.8 && python3 -m venv myenv && source myenv/bin/activate && python3 -m pip install --no-cache-dir vllm" >&2
    exit 1
fi

echo "Python env:  $ACTIVATED_ENV ($(command -v python3))"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1

echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-n/a}"
echo "Prompt type: $PROMPT_TYPE"
echo "Input CSV:   $INPUT_CSV"
echo "Output CSV:  $OUTPUT_CSV"
echo "Run dir:     $RUN_DIR"
echo "Model:       $MODEL"

INFER_SCRIPT="$RUN_DIR/batch_infer_simple.py"
PROMPT_MD="$RUN_DIR/prompt_examples.md"

srun python3 "$INFER_SCRIPT" \
    --input-csv "$INPUT_CSV" \
    --text-column text \
    --output-csv "$OUTPUT_CSV" \
    --prompt-md "$PROMPT_MD" \
    --prompt-type "$PROMPT_TYPE" \
    --model "$MODEL" \
    --runtime-profile a100 \
    --gpu-mem-util 0.94 \
    --max-model-len 1024 \
    --max-num-seqs 64 \
    --batch-size 64 \
    --max-tokens 96 \
    --context-reserve-tokens 128 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --resume \
    --truncate-to-fit \
    --flush-every 8 \
    --log-every 5000 \
    "${EXTRA_ARGS[@]}"
