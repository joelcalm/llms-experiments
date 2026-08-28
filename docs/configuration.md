# Configuration Contract

One YAML file describes one run. `${ENV_VAR}` references and dotted
`--set key=value` overrides are resolved before validation. Required sections
are `run`, `model`, `variants`, `output`, and exactly one of `input` or
`datasets`.

By default, relative asset, input, log, and output paths are resolved from the
configuration's directory (or the repository root for files directly under
`config/` or `experiments/`). A nested configuration can set an explicit
relative `config_root`, resolved from the YAML file itself:

```yaml
config_root: ../..
```

This keeps a self-contained configuration family under a subdirectory while
sharing repository-level prompt assets and the usual `results/` location.

## Inputs and backends

An input requires `path`, `format`, `id_column`, and `text_column`. Supported
formats are `csv`, `tsv`, `jsonl`, `parquet`, `nested_json`, and `paired_tsv`.
Every reader emits a deterministic `_source_position`; optional gold labels are
normalized into `_gold_labels`.

`model.backend` is one of `local_vllm`, `local_vllm_async`,
`openai_compatible`, `llama_cpp`, or `fake`. The CLI accepts the aliases
`vllm`, `vllm-async`, `openai-compatible`, and `llama-cpp` and maps them to
those configuration values. Backend-specific keys remain in `model`; they do
not affect processor configuration. The optional `llama-cpp` package is
required only when `llama_cpp` is selected, and the optional `gpu` extra is
required for either vLLM backend.

For local vLLM, structured JSON generation is enabled by default whenever a
`json_schema` stage is present. A model-specific runtime that cannot safely
combine its tokenizer or output protocol with vLLM guided decoding may set
`model.structured_outputs: false`; the processor still parses and validates
the returned text, so this option removes only generation-time constraints.

The optional `model.kv_cache_dtype` is passed directly to vLLM. `auto` keeps
the model's native cache type; on CUDA 11.8 or newer, `fp8` (or the explicit
`fp8_e4m3`/`fp8_e5m2` spelling supported by the installed vLLM version) stores
the KV cache in FP8 and can increase the number of resident tokens. The
setting is recorded in the effective configuration and manifest because it
can affect both memory behaviour and numerical outputs. Use a small,
representative smoke run before changing a completed comparison, and keep
calibrated KV scales when the checkpoint provides them. The ProtoEthos worker
accepts the corresponding `KV_CACHE_DTYPE` environment variable; when it is
unset, no override is added.

## Variants and processors

Each variant has an `id`, one or more Markdown `prompts`, and a processor:

```yaml
variants:
  - id: single_label
    max_tokens: 64
    prompts:
      - prompts/task.md
      - prompts/input.md
    processor:
      result: single_label
      stages:
        - type: json_decode
        - type: json_schema
          schema: schemas/single-label.json
          enum_from: dataset_labels
```

`processor.result` is the semantic value written to `result_type`.
`processor.stages` is ordered and must be non-empty. A processor omitted from a
variant uses one `identity` stage and emits a visible warning. The removed
fields `request_mode`, `result_type`, and `expand_over` are ignored with visible
migration warnings; new configurations must use processor stages.

### Stage configuration

| Stage | Important keys | Effect |
| --- | --- | --- |
| `identity` | none | returns raw text; must be the only stage |
| `fan_out` | `over: dataset_labels` | creates one stable request row per source row and label |
| `json_decode` | none | parses response text as JSON |
| `json_schema` | `schema`, optional `enum_from: dataset_labels` | supplies structured-generation constraints and validates the parsed value |
| `candidate_logprobs` | `candidates` or `candidates_from` | requests one token and aggregates candidate log probabilities |
| `verbalized_confidence` | optional `top_logprobs`, digit field names | enriches validated JSON using separate tens/units token distributions |

`candidates_from` is `dataset_labels` or `code_labels`. Candidate extraction
requests enough headroom for tokenizer spellings, capped at 20 alternatives.
`verbalized_confidence.top_logprobs` must be between 10 and 20.

The processor is compiled once per variant. It merges stage-declared generation
requirements into one backend-neutral request, so the same pipeline is used by
in-process backends, OpenAI-compatible endpoints, and external batch parsing.
Row-preparation stages (currently `fan_out`) must precede response stages.

For per-label scoring, fan-out is explicit:

```yaml
processor:
  result: label_yes_no_logprobs
  stages:
    - type: fan_out
      over: dataset_labels
    - type: candidate_logprobs
      candidates: ["yes", "no"]
```

## Outputs

`output.directory` is required. `output.format` is `parquet`, `csv`, or
`jsonl`; omitting it selects Parquet with a visible warning. One format is used
for parts, the combined result, and per-variant projections:

```yaml
output:
  directory: outputs/my-run
  format: jsonl
  include_text: false
  include_raw_response: true
```

All formats preserve the same logical result schema and resume semantics.
For a matrix split across concurrent jobs, set `output.variant_isolated: true`
and configure exactly one variant per job. The runner then uses the
variant-specific `resume_index_name`, `manifest_name`, and
`effective_config_name`, writes only that variant projection, and avoids
concurrent writes to the combined result. `matrix_manifest_name` and
`logging.suffix` can likewise keep matrix summaries and event logs separate.
The normal `matrix_manifest.json`, `manifest.json`, `effective_config.yaml`,
and combined result are retained for single-process runs; a later consolidation
step can rebuild the combined result from the isolated variant projections.

## Runtime controls

`resources.cpu` controls affinity, reserved cores, and native thread pools.
`batch` controls fixed/adaptive candidates, warm-up size, minimum size, and
halving after batch failures. With `batch.dynamic_limit: true`, the runner
doubles `batch.start_size` over the currently materialised lane rows. If the
largest accepted probe reaches the end of that window, it materialises more
rows from the same lane and continues doubling. The search stops only at the
first backend capacity failure or when the lane has no more rows. It records
each probe and every window extension in the lane event log; the Condor worker
can restart a dead vLLM process at a conservative accepted size. Thus,
`batch.warmup_rows` seeds the search but is not a batch-size ceiling.
For a backend known to remain usable after a failed capacity probe,
`batch.refine_after_failure: true` performs integer bisection between the last
accepted and first rejected probes. This can select a non-power-of-two batch
near the available-memory boundary; leave it false for backends that require
restart after an OOM. The ProtoEthos FP8-KV-cache smoke/recovery jobs enable
it only after the corresponding model has passed the smoke gate.
`batch.queue_depth` may make each backend call larger than `model.max_num_seqs`
so vLLM can refill finished sequence slots while slower rows are still
decoding. Use it only after a measured GPU smoke test, because the resulting
chunk still consumes host memory and output buffering.

The ProtoEthos Condor worker captures the complete runner output for every
failed invocation. If an OOM occurs during startup or batch probing, the
status record is written as `inference_failed` with `failure_kind: oom` and a
`failure_log` path, then the worker exits so Condor releases the single GPU.
More specific fatal signatures take precedence: incompatible quantisation
kernels are recorded as `kernel_incompatible` rather than being retried as an
OOM. A failed dynamic probe with accepted batches is resumed at the preceding
accepted size (or half the sole accepted size); a startup failure with no
accepted batch is not allowed to hold a GPU while waiting for another resource.

`validation.retry` is bounded and can defer structured-response retries into
smaller batches with a larger token budget. By default, a durable
`failed_validation` row is treated as complete on resume. Set
`retry_failed_validation_on_resume: true` when a corrected prompt, schema, or
token budget is available; the next run then replaces those rows while
preserving the normal default for deterministic validation failures. The
ProtoEthos full-matrix configuration enables this opt-in because its compact
release requires every valid row.
`streaming.output_chunk_rows` bounds unpublished output rows.

For a long homogeneous classification lane, `local_vllm_async` together with
`streaming.dispatch: async_vllm` submits individual, lazily read prompts to
vLLM's continuous scheduler. `streaming.max_in_flight` is a bounded producer
queue, not an engine batch size: vLLM still enforces `model.max_num_seqs` and
`model.max_num_batched_tokens`. A useful starting point is two to four times
the measured safe sequence count, followed by a representative throughput
smoke test. Each `streaming.output_chunk_rows` final responses is atomically
published as a part, then its resume keys are committed, then
`output.checkpoint_name` (default `checkpoint.json`) is atomically refreshed.
On `SIGTERM`/`SIGINT`, the runner flushes the current completed-response buffer
before terminating; work still inside the engine is intentionally retried from
the last durable part. This provides bounded-memory streaming without claiming
that an uncompleted request was persisted.
One persistent event loop is retained for every strategy and dataset sharing an
`AsyncLLM` engine; it is not recreated between lanes. Backend failures publish
the same durable boundary with checkpoint state `failed`, while orderly
interruptions use `interrupted`.

For hybrid recurrent models, `model.mamba_cache_dtype` and
`model.mamba_ssm_cache_dtype` are passed through to vLLM when supported. Keep
those settings, `model.kv_cache_dtype`, `model.performance_mode`,
`model.optimization_level`, `model.language_model_only`, and any
`VLLM_USE_FASTOKENS` environment setting in the effective configuration and
manifest. They are runtime conditions that may affect both throughput and the
numerical comparison, so validate them on a fixed stratified subset before
using them for a new reported cohort.

The machine-readable schema is [config.schema.json](config.schema.json).
Regenerate it with `uv run python scripts/generate_config_schema.py` after changing
the typed models.

The schema is **partial**: it validates only the typed Pydantic models in
`ExperimentConfig`. Several backend/runtime passthrough keys are legal because the
relevant sections use `ConfigSection(extra="allow")` and are forwarded verbatim to
the backend or runner. These include, among others: `model.kv_cache_dtype`,
`model.mamba_cache_dtype`, `model.performance_mode`, `model.optimization_level`,
`model.language_model_only`, `output.variant_isolated`, `output.resume_index_name`/
`manifest_name`/`effective_config_name`/`matrix_manifest_name`/`checkpoint_name`,
`streaming.dispatch`, `batch.dynamic_limit`, `batch.refine_after_failure`, and
`batch.queue_depth`. They are consumed by `orchestration.py` and the backends even
though they are absent from the schema; treat the schema as a structural guide, not
an exhaustive allow-list.
