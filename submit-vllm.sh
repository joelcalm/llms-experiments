#!/bin/bash
#SBATCH --job-name=vllm-scores
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch submit-vllm.sh [PROMPT_TYPE] [SCORE_MAX] [INPUT_CSV] [OUTPUT_CSV] [MODEL] [extra batch_infer_simple.py args...]
# Example:
#   sbatch submit-vllm.sh MFT 20 v2_3m_final_clean_text.csv outputs/mft_scores_0_20.csv qwen-2b-awq

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

if [[ ! -f "$RUN_DIR/prompts/prompt_examples_0_100.md" ]]; then
    echo "ERROR: Missing $RUN_DIR/prompts/prompt_examples_0_100.md" >&2
    exit 2
fi

if [[ ! -f "$RUN_DIR/prompts/prompt_examples_0_20.md" ]]; then
    echo "ERROR: Missing $RUN_DIR/prompts/prompt_examples_0_20.md" >&2
    exit 2
fi

if [[ ! -f "$RUN_DIR/prompts/prompt_examples_0_5.md" ]]; then
    echo "ERROR: Missing $RUN_DIR/prompts/prompt_examples_0_5.md" >&2
    exit 2
fi

PROMPT_TYPE="${1:-MFT}"
SCORE_MAX="${2:-100}"
INPUT_CSV="${3:-v2_3m_final_clean_text.csv}"
OUTPUT_CSV="${4:-$RUN_DIR/outputs/${PROMPT_TYPE,,}_scores_0_${SCORE_MAX}_1m_seed20260417_${SLURM_JOB_ID:-local}.csv}"
MODEL="${5:-qwen-2b-awq}"

if [[ $# -ge 5 ]]; then
    shift 5
else
    shift $#
fi
EXTRA_ARGS=("$@")

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs"

case "$PROMPT_TYPE" in
    MFT|SHVT)
        ;;
    *)
        echo "ERROR: PROMPT_TYPE must be MFT or SHVT (got '$PROMPT_TYPE')" >&2
        exit 2
        ;;
esac

case "$SCORE_MAX" in
    100)
        PROMPT_MD_DEFAULT="$RUN_DIR/prompts/prompt_examples_0_100.md"
        ;;
    20)
        PROMPT_MD_DEFAULT="$RUN_DIR/prompts/prompt_examples_0_20.md"
        ;;
    10)
        PROMPT_MD_DEFAULT="$RUN_DIR/prompts/prompt_examples_0_10.md"
        ;;
    5)
        PROMPT_MD_DEFAULT="$RUN_DIR/prompts/prompt_examples_0_5.md"
        ;;
    *)
        echo "ERROR: SCORE_MAX must be one of: 100, 20, 10, 5 (got '$SCORE_MAX')" >&2
        exit 2
        ;;
esac

PROMPT_MD="${PROMPT_MD_OVERRIDE:-$PROMPT_MD_DEFAULT}"

SAMPLE_SIZE="${SAMPLE_SIZE:-1000000}"
SAMPLE_SEED="${SAMPLE_SEED:-20260417}"
SAMPLED_CSV="${SAMPLED_CSV:-$RUN_DIR/outputs/sample_${SAMPLE_SIZE}_seed${SAMPLE_SEED}.csv}"
SAMPLER_SCRIPT="$RUN_DIR/sample_csv_deterministic.py"

# Throughput tuning defaults for A100 + qwen-2b-awq.
# Override at submit time with env vars, e.g. BATCH_SIZE=128 MAX_NUM_SEQS=128 sbatch ...
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
BATCH_SIZE="${BATCH_SIZE:-96}"
MAX_TOKENS="${MAX_TOKENS:-96}"
CONTEXT_RESERVE_TOKENS="${CONTEXT_RESERVE_TOKENS:-128}"
FLUSH_EVERY="${FLUSH_EVERY:-16}"
LOG_EVERY="${LOG_EVERY:-10000}"

# vLLM compile path has been unstable in this environment; eager mode is safer.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_PREFIX_CACHING="${VLLM_PREFIX_CACHING:-0}"

RUNTIME_EXTRA_ARGS=()
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
    RUNTIME_EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "$VLLM_PREFIX_CACHING" == "1" ]]; then
    RUNTIME_EXTRA_ARGS+=(--enable-prefix-caching)
else
    RUNTIME_EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

if [[ ! -f "$SAMPLER_SCRIPT" ]]; then
    echo "ERROR: Missing sampler script: $SAMPLER_SCRIPT" >&2
    exit 2
fi

# CESGA typically expects software loaded via spack. Keep these optional so the script
# still works if your environment is already activated elsewhere.
if command -v spack >/dev/null 2>&1; then
    spack load python@3.11 || true
    spack load cuda@12.4 || true
fi

ACTIVATED_ENV="myenv"
ENV_PYTHON="$RUN_DIR/$ACTIVATED_ENV/bin/python3"
if [[ -x "$ENV_PYTHON" ]]; then
    if ! "$ENV_PYTHON" -c "import vllm" >/dev/null 2>&1; then
        echo "ERROR: 'myenv' was found but cannot import 'vllm'." >&2
        echo "Reinstall in myenv: module load cesga/2022 python/3.10.8 && python3 -m venv myenv && myenv/bin/python3 -m pip install --no-cache-dir vllm" >&2
        exit 1
    fi
else
    echo "ERROR: Missing $ENV_PYTHON" >&2
    echo "Create myenv: module load cesga/2022 python/3.10.8 && python3 -m venv myenv && myenv/bin/python3 -m pip install --no-cache-dir vllm" >&2
    exit 1
fi

echo "Python env:  $ACTIVATED_ENV ($ENV_PYTHON)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "$SAMPLED_CSV" ]]; then
    SAMPLE_LOCK_DIR="${SAMPLED_CSV}.lock"
    if mkdir "$SAMPLE_LOCK_DIR" 2>/dev/null; then
        trap 'rmdir "$SAMPLE_LOCK_DIR" >/dev/null 2>&1 || true' EXIT
        echo "Building deterministic sample: size=$SAMPLE_SIZE seed=$SAMPLE_SEED"
        "$ENV_PYTHON" "$SAMPLER_SCRIPT" \
            --input-csv "$INPUT_CSV" \
            --output-csv "$SAMPLED_CSV" \
            --text-column text \
            --sample-size "$SAMPLE_SIZE" \
            --seed "$SAMPLE_SEED"
        rmdir "$SAMPLE_LOCK_DIR" >/dev/null 2>&1 || true
        trap - EXIT
    else
        echo "Waiting for existing sample build to finish: $SAMPLED_CSV"
        while [[ ! -f "$SAMPLED_CSV" && -d "$SAMPLE_LOCK_DIR" ]]; do
            sleep 30
        done
    fi
fi

if [[ ! -f "$SAMPLED_CSV" ]]; then
    echo "ERROR: Expected sampled CSV was not created: $SAMPLED_CSV" >&2
    exit 2
fi

echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURMD_NODENAME:-n/a}"
echo "Prompt type: $PROMPT_TYPE"
echo "Score max:   $SCORE_MAX"
echo "Input CSV:   $SAMPLED_CSV"
echo "Output CSV:  $OUTPUT_CSV"
echo "Run dir:     $RUN_DIR"
echo "Model:       $MODEL"
echo "Prompt file: $PROMPT_MD"
echo "Sample seed: $SAMPLE_SEED"
echo "Runtime:     gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS batch_size=$BATCH_SIZE"
echo "vLLM mode:   enforce_eager=$VLLM_ENFORCE_EAGER prefix_caching=$VLLM_PREFIX_CACHING"

INFER_SCRIPT="$RUN_DIR/batch_infer_simple.py"

srun "$ENV_PYTHON" "$INFER_SCRIPT" \
    --input-csv "$SAMPLED_CSV" \
    --text-column text \
    --output-csv "$OUTPUT_CSV" \
    --prompt-md "$PROMPT_MD" \
    --prompt-type "$PROMPT_TYPE" \
    --score-max "$SCORE_MAX" \
    --model "$MODEL" \
    --runtime-profile a100 \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --context-reserve-tokens "$CONTEXT_RESERVE_TOKENS" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    "${RUNTIME_EXTRA_ARGS[@]}" \
    --resume \
    --truncate-to-fit \
    --flush-every "$FLUSH_EVERY" \
    --log-every "$LOG_EVERY" \
    --definitions \
    "${EXTRA_ARGS[@]}"
