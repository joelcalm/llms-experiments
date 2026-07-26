#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=configs/protoethos/ministral_all_datasets.yaml}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"
: "${MODEL_ID:?Set MODEL_ID to the model identifier for this job}"

args=(llms-experiments run "$CONFIG" --output "$OUTPUT_DIR" --model "$MODEL_ID")
add_model_override() {
    local environment_name="$1"
    local config_key="$2"
    local value="${!environment_name:-}"
    if [[ -n "$value" && "$value" != "NONE" ]]; then
        args+=(--set "$config_key=$value")
    fi
}

add_model_override GPU_MEMORY_UTILIZATION model.gpu_memory_utilization
add_model_override MAX_NUM_SEQS model.max_num_seqs
add_model_override MM_ENCODER_ATTN_BACKEND model.mm_encoder_attn_backend
add_model_override ATTN_BACKEND model.attention_backend
add_model_override MOE_BACKEND model.moe_backend
if [[ -n "${DATASETS:-}" ]]; then
    args+=(--datasets "$DATASETS")
fi
if [[ -n "${VARIANTS:-}" ]]; then
    args+=(--variants "$VARIANTS")
fi

export HF_HOME="${HF_HOME:-${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/hf_cache}"
export TOKENIZERS_PARALLELISM=false
exec "${args[@]}"
