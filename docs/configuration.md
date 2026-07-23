# Configuration Contract

One YAML file describes one run. `${ENV_VAR}` references are expanded before validation; missing variables fail clearly. Dotted `--set key=value` overrides are parsed as YAML scalars and applied before typed validation.

Required sections are `run`, `model`, `variants`, `output`, and exactly one of `input` or `datasets`. Dataset matrices contain entries with a stable `id` and an `input` mapping. IDs for datasets and variants must be unique.

`input` requires `path`, `format`, `id_column`, and `text_column`. Supported readers include CSV, JSON, nested JSON, JSONL, paired TSV, and Parquet. Optional label columns are normalised against the configured taxonomy. Streaming mode iterates bounded batches rather than loading the complete source into memory.

`model.backend` is `local_vllm`, `openai_compatible`, or `fake`.
OpenAI-compatible endpoints use configurable `api_base_url` and `api_key_env`; no specific vendor is assumed. Local vLLM options include model length, maximum sequences, memory utilisation, tensor parallelism, prefix caching, and language-model-only mode.

On SM 12.x devices, the runner automatically selects vLLM's non-FlashInfer sampler fallback when no explicit `VLLM_USE_FLASHINFER_SAMPLER` setting is provided. This avoids FlashInfer sampler JIT incompatibilities while leaving normal GPU attention backends intact.

Every variant requires `id`, `request_mode`, `result_type`, and one or more Markdown prompt paths. Request mode is `generate`, `generate_with_logprobs`, or `candidate_logprobs`. `generate_with_logprobs` retains the top token distribution at every generated position; its `top_logprobs` parameter must be between 10 and the common maximum of 20. Semantic result types are documented in the result contract and are independent of variant names. Structured variants may reference a JSON Schema. Label-wise variants use `expand_over: dataset_labels`.

`resources.cpu` controls affinity, reserved cores, and native thread-pool limits. `batch` controls fixed/adaptive candidates, warm-up size, minimum size, and halving on failure. `validation.retry` is bounded and may defer retries at a smaller batch size. `streaming.output_chunk_rows` bounds unpublished rows.

The machine-readable schema source is [config.schema.json](config.schema.json). Run `python scripts/generate_config_schema.py` after changing typed models.
