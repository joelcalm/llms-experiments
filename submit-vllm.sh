#!/bin/bash
#SBATCH --job-name=vllm-one-label
#SBATCH --time=20:00:00
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
#
# Defaults:
#   PROMPT_TYPE=MFT
#   INPUT_CSV=protoethosv2_3m_finalqc_20260512.csv
#   SAMPLE_SIZE=100000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CODE_DIR: where the runnable project lives: batch_infer_simple.py, sample_csv_deterministic.py, myenv, etc.
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$ROOT_DIR}}"

# STORE_DIR: where logs, outputs, samples and prompts live.
STORE_DIR="${STORE_DIR:-$CODE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$STORE_DIR/outputs}"
LOG_DIR="${LOG_DIR:-$STORE_DIR/logs}"

# Prompt files can be either in $STORE_DIR/prompts or directly in $STORE_DIR.
PROMPT_DIR="${PROMPT_DIR:-$STORE_DIR/prompts}"

cd "$CODE_DIR"

if [[ ! -d "$CODE_DIR" ]]; then
    echo "ERROR: Code dir does not exist: $CODE_DIR" >&2
    exit 2
fi

if [[ ! -d "$STORE_DIR" ]]; then
    echo "ERROR: Store dir does not exist: $STORE_DIR" >&2
    exit 2
fi

if [[ ! -f "$CODE_DIR/batch_infer_simple.py" ]]; then
    echo "ERROR: Missing $CODE_DIR/batch_infer_simple.py" >&2
    echo "Hint: CODE_DIR must point to the project root containing batch_infer_simple.py." >&2
    exit 2
fi

SAMPLE_SEED="${SAMPLE_SEED:-20260417}"
PROMPT_TYPE="${1:-MFT}"
INPUT_CSV="${2:-$CODE_DIR/protoethosv2_3m_finalqc_20260512.csv}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100000}"
RUN_NAME="${RUN_NAME:-${PROMPT_TYPE,,}_one_label_${SAMPLE_SIZE}_seed${SAMPLE_SEED}_${SLURM_JOB_ID:-local}}"
OUTPUT_CSV="${3:-$OUTPUT_DIR/${RUN_NAME}.csv}"
MODEL="${4:-qwen-2b-awq}"
if [[ $# -ge 4 ]]; then
    shift 4
else
    shift $#
fi
EXTRA_ARGS=("$@")

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

case "$PROMPT_TYPE" in
    MFT|SHVT)
        ;;
    *)
        echo "ERROR: PROMPT_TYPE must be MFT or SHVT (got '$PROMPT_TYPE')" >&2
        exit 2
        ;;
esac

case "$PROMPT_TYPE" in
    MFT)
        PROMPT_FILE_NAME="mft_one_label.txt"
        ;;
    SHVT)
        PROMPT_FILE_NAME="shvt_one_label.txt"
        ;;
esac

PROMPT_MD_DEFAULT="$PROMPT_DIR/$PROMPT_FILE_NAME"

# Fallback: if prompts/ does not contain the file, try directly in STORE_DIR.
if [[ ! -f "$PROMPT_MD_DEFAULT" && -f "$STORE_DIR/$PROMPT_FILE_NAME" ]]; then
    PROMPT_MD_DEFAULT="$STORE_DIR/$PROMPT_FILE_NAME"
fi

PROMPT_MD="${PROMPT_MD_OVERRIDE:-$PROMPT_MD_DEFAULT}"

if [[ ! -f "$PROMPT_MD" ]]; then
    echo "ERROR: Missing prompt file: $PROMPT_MD" >&2
    echo "Checked PROMPT_DIR=$PROMPT_DIR and STORE_DIR=$STORE_DIR" >&2
    exit 2
fi

SAMPLED_CSV="${SAMPLED_CSV:-$OUTPUT_DIR/sample_${PROMPT_TYPE,,}_one_label_${SAMPLE_SIZE}_seed${SAMPLE_SEED}.csv}"
SAMPLER_SCRIPT="$CODE_DIR/sample_csv_deterministic.py"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
BATCH_SIZE="${BATCH_SIZE:-96}"
MAX_TOKENS="${MAX_TOKENS:-96}"
CONTEXT_RESERVE_TOKENS="${CONTEXT_RESERVE_TOKENS:-128}"
FLUSH_EVERY="${FLUSH_EVERY:-16}"
LOG_EVERY="${LOG_EVERY:-10000}"

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

SAMPLE_TEXT_COLUMN="sentence"
SAMPLER_EXTRA_ARGS=(--filter-column theory_id --filter-value "${PROMPT_TYPE,,}")
INFER_ARGS=(
    --text-column sentence
    --id-column id
    --label-column value_id
    --sample-index-column sample_index
    --metrics-run-name "$RUN_NAME"
)

if command -v spack >/dev/null 2>&1; then
    spack load python@3.11 || true
    spack load cuda@12.4 || true
fi

ACTIVATED_ENV="myenv"
ENV_PYTHON="$CODE_DIR/$ACTIVATED_ENV/bin/python3"

if [[ -x "$ENV_PYTHON" ]]; then
    if ! "$ENV_PYTHON" -c "import vllm" >/dev/null 2>&1; then
        echo "ERROR: 'myenv' was found but cannot import 'vllm'." >&2
        echo "Reinstall in myenv: module load cesga/2022 python/3.10.8 && python3 -m venv myenv && myenv/bin/python3 -m pip install --no-cache-dir vllm" >&2
        exit 1
    fi
else
    echo "ERROR: Missing $ENV_PYTHON" >&2
    echo "Create myenv inside CODE_DIR or set CODE_DIR correctly." >&2
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
            --text-column "$SAMPLE_TEXT_COLUMN" \
            --sample-size "$SAMPLE_SIZE" \
            --seed "$SAMPLE_SEED" \
            "${SAMPLER_EXTRA_ARGS[@]}"

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

echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         ${SLURMD_NODENAME:-n/a}"
echo "Prompt type:  $PROMPT_TYPE"
echo "Code dir:     $CODE_DIR"
echo "Store dir:    $STORE_DIR"
echo "Input CSV:    $SAMPLED_CSV"
echo "Output CSV:   $OUTPUT_CSV"
echo "Model:        $MODEL"
echo "Prompt file:  $PROMPT_MD"
echo "Sample size:  $SAMPLE_SIZE"
echo "Sample seed:  $SAMPLE_SEED"
echo "Runtime:      gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS batch_size=$BATCH_SIZE"
echo "vLLM mode:    enforce_eager=$VLLM_ENFORCE_EAGER prefix_caching=$VLLM_PREFIX_CACHING"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

INFER_SCRIPT="$CODE_DIR/batch_infer_simple.py"

srun "$ENV_PYTHON" "$INFER_SCRIPT" \
    --input-csv "$SAMPLED_CSV" \
    --output-csv "$OUTPUT_CSV" \
    --prompt-md "$PROMPT_MD" \
    --prompt-type "$PROMPT_TYPE" \
    "${INFER_ARGS[@]}" \
    --model "$MODEL" \
    --runtime-profile a100 \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --context-reserve-tokens "$CONTEXT_RESERVE_TOKENS" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --dtype bfloat16 \
    "${RUNTIME_EXTRA_ARGS[@]}" \
    --resume \
    --truncate-to-fit \
    --flush-every "$FLUSH_EVERY" \
    --log-every "$LOG_EVERY" \
    "${EXTRA_ARGS[@]}"
