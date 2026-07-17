#!/usr/bin/env bash
# Run one full experiment-cli matrix on a single H100 slot.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
MODEL_SLUG="${MODEL_SLUG:?MODEL_SLUG is required}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION is required}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:?MAX_NUM_SEQS is required}"
# Optional: comma-separated dataset ids to restrict this run to. A dataset
# left out is never touched by run()'s resume/invalidation logic at all, so
# this is how an already-correct dataset survives a config fix untouched
# instead of being swept up by the shared resume fingerprint of the ones
# that actually need to be redone.
DATASETS="${DATASETS:-}"
# Optional: force vLLM's ViT/multimodal-encoder attention backend. Some
# models (e.g. the Qwen3.5/3.6 family) initialize a vision-encoder attention
# module even under language_model_only=true, and their default backend
# selection picks FlashAttention without actually checking whether the
# installed wheel supports this GPU's compute capability, crash-looping
# instead of failing cleanly. TORCH_SDPA routes through PyTorch's native
# scaled_dot_product_attention (memory-efficient kernel, hardware-neutral)
# instead of the external flash-attn wheel.
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-}"
if [[ "$MM_ENCODER_ATTN_BACKEND" == "NONE" ]]; then
    MM_ENCODER_ATTN_BACKEND=""
fi

CODE_DIR="$RUN_ROOT/code"
PROJECT_DIR="$CODE_DIR"
ENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
OUTPUT_DIR="$RUN_ROOT/results/$MODEL_SLUG"
STATUS_DIR="$RUN_ROOT/status"

mkdir -p "$OUTPUT_DIR" "$STATUS_DIR"

# The data tree is staged underneath RUN_ROOT; the YAML remains portable by
# resolving its existing ${TFM_ROOT} paths against this directory.
export TFM_ROOT="$RUN_ROOT/TFM"
# Model weights and JIT-compile artefacts are downloaded/built fresh on
# whichever worker node runs the job, but persisted on shared cluster storage
# (not this job's node-local scratch, which is wiped on exit, and not $HOME,
# whose small quota can EDQUOT mid-compile). This lets retries and later runs
# reuse the same downloads/compiled kernels instead of repeating multi-minute,
# multi-gigabyte work every restart. Defaults sit under RUN_ROOT; export
# SHARED_CACHE_ROOT to point at a persistent cache shared across runs.
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-$RUN_ROOT/shared_cache}"
# snapshot_download verifies files by hash/ETag, so pointing HF_HOME at an
# existing cache reuses whatever is already complete and only fetches what is
# missing or partial. Export HF_HOME to reuse a cache that already holds the
# target checkpoints.
export HF_HOME="${HF_HOME:-$RUN_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export TORCHINDUCTOR_CACHE_DIR="$SHARED_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$SHARED_CACHE_ROOT/triton"
export XDG_CACHE_HOME="$SHARED_CACHE_ROOT/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

# Condor assigns this slot's GPU as a UUID (e.g. GPU-xxxxxxxx). vLLM's model
# registry inspection subprocess calls int(CUDA_VISIBLE_DEVICES) and crashes
# on that form, so resolve it to the equivalent numeric index; the process
# still only sees this one physical GPU either way.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    resolved_index="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
        | awk -F', ' -v uuid="$CUDA_VISIBLE_DEVICES" '$2==uuid{print $1}')"
    if [[ -n "$resolved_index" ]]; then
        export CUDA_VISIBLE_DEVICES="$resolved_index"
    fi
fi

started_at="$(date --iso-8601=seconds)"
printf '{"model":"%s","status":"started","started_at":"%s"}\n' \
    "$MODEL_ID" "$started_at" > "$STATUS_DIR/$MODEL_SLUG.json"

if [[ ! -x "$ENV_PYTHON" ]]; then
    printf '{"model":"%s","status":"missing_environment"}\n' "$MODEL_ID" > "$STATUS_DIR/$MODEL_SLUG.json"
    echo "Missing $ENV_PYTHON; build the RUN_ROOT environment before submitting workers." >&2
    exit 2
fi

printf '{"model":"%s","status":"downloading_model","cache":"%s"}\n' \
    "$MODEL_ID" "$HF_HOME" > "$STATUS_DIR/$MODEL_SLUG.json"
if ! "$ENV_PYTHON" -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["MODEL_ID"])'; then
    printf '{"model":"%s","status":"model_download_failed"}\n' "$MODEL_ID" > "$STATUS_DIR/$MODEL_SLUG.json"
    exit 3
fi

datasets_args=()
if [[ -n "$DATASETS" ]]; then
    datasets_args=(--datasets "$DATASETS")
fi

mm_encoder_args=()
if [[ -n "$MM_ENCODER_ATTN_BACKEND" ]]; then
    mm_encoder_args=(--set "model.mm_encoder_attn_backend=$MM_ENCODER_ATTN_BACKEND")
fi

"$ENV_PYTHON" "$CODE_DIR/experiment-cli/experiment_cli.py" run-matrix \
    --config "$CODE_DIR/experiments/ministral_all_datasets.yaml" \
    --run-id "$MODEL_SLUG" \
    --model "$MODEL_ID" \
    --output "$OUTPUT_DIR" \
    "${datasets_args[@]}" \
    "${mm_encoder_args[@]}" \
    --set "model.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
    --set 'model.max_model_len=1024' \
    --set "model.max_num_seqs=$MAX_NUM_SEQS" \
    --set 'model.tokenizer_mode=null' \
    --set 'model.config_format=null' \
    --set 'model.load_format=null' \
    --set 'model.tensor_parallel_size=1' \
    --set 'model.dtype=bfloat16' \
    --set 'model.language_model_only=true' \
    --set 'model.chat_template_kwargs.enable_thinking=false' \
    --set 'model.enable_prefix_caching=true' \
    --set 'model.trust_remote_code=true' \
    --set 'resources.cpu.cores=16' \
    --set 'resources.cpu.reserve_cores=0' \
    --set 'resources.cpu.thread_pool_size=1' \
    --set 'batch.candidates=[1,2,4,8,16,32,64,128]' \
    --set 'batch.warmup_rows=64'

finished_at="$(date --iso-8601=seconds)"
printf '{"model":"%s","status":"completed","started_at":"%s","finished_at":"%s"}\n' \
    "$MODEL_ID" "$started_at" "$finished_at" > "$STATUS_DIR/$MODEL_SLUG.json"
