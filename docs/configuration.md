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

`model.backend` is one of `local_vllm`, `openai_compatible`, `llama_cpp`, or
`fake`. The CLI accepts the aliases `vllm`, `openai-compatible`, and
`llama-cpp` and maps them to those configuration values. Backend-specific keys
remain in `model`; they do not affect processor configuration. The optional
`llama-cpp` package is required only when `llama_cpp` is selected, and the
optional `gpu` extra is required for `local_vllm`.

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
| `identity` | optional `max_tokens` | returns raw text; must be the only stage |
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

## Runtime controls

`resources.cpu` controls affinity, reserved cores, and native thread pools.
`batch` controls fixed/adaptive candidates, warm-up size, minimum size, and
halving after batch failures. `validation.retry` is bounded and can defer
structured-response retries into smaller batches with a larger token budget.
`streaming.output_chunk_rows` bounds unpublished output rows.

The machine-readable schema is [config.schema.json](config.schema.json).
Regenerate it with `python scripts/generate_config_schema.py` after changing
the typed models.
