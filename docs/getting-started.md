# Getting started

This guide is the shortest path from a fresh checkout to understanding and
modifying the library.

## 1. Install and run the offline check

The default installation has no CUDA dependency and is sufficient for config
validation, the fake backend, external-batch preparation/parsing, and tests:

```bash
uv sync --all-groups
uv run llms-experiments validate experiments/matrix_smoke.yaml
uv run llms-experiments run experiments/matrix_smoke.yaml --backend fake --rows 2
```

The fake backend makes deterministic responses, so this check does not need a
model, GPU, endpoint, or API key. Use `uv run llms-experiments doctor` to
inspect the local package and GPU/driver visibility.

## Repository overview

This repository is an inference and experiment-execution framework, not a
model-training repository. The configuration describes the complete run:
input data, model backend, Markdown prompt templates, processor stages,
resource and retry controls, and output format.

The run lifecycle is:

```text
YAML configuration
  -> input reader and normalized rows
  -> prompt rendering and processor preparation
  -> backend generation
  -> semantic result processing
  -> durable Parquet, CSV, or JSONL output
```

The four main concerns are independent:

1. **Input readers** load CSV, TSV, JSONL, Parquet, nested JSON, or paired TSV
   data and preserve a deterministic `_source_position`.
2. **Backends** execute requests through vLLM, llama.cpp,
   OpenAI-compatible endpoints, or the deterministic `fake` backend. They
   return raw text and token evidence; they do not interpret labels.
3. **Processors** own semantic interpretation: parsing, schema validation,
   fan-out, candidate log probabilities, confidence extraction, and result
   construction.
4. **Output stores** write the common Result Contract 2.0 to Parquet, CSV, or
   JSONL, including atomic parts and SQLite-backed resume state.

The same processor is used by in-process inference and external batch parsing.
This is the key design rule to preserve when extending the system: backend
transport and result semantics must remain separate.

Every run records a manifest, effective configuration, provenance hashes, row
statuses, and resource diagnostics. Interrupted runs can resume from durable
rows without duplicating result identities. Generated outputs and model
artifacts should not be committed to Git.

## 2. Understand one run

There are four independent concerns:

1. An **input reader** turns CSV, TSV, JSONL, Parquet, nested JSON, or paired
   TSV data into normalized rows with a stable `_source_position`.
2. A **processor** prepares rows, declares the evidence it needs, and converts
   each raw response into a semantic `ProcessedResult`.
3. A **backend** executes the compiled `GenerationRequest` and returns
   backend-neutral text/token evidence. It does not parse labels or calculate
   confidence.
4. An **output store** writes the common result contract to Parquet, CSV, or
   JSONL, including atomic parts and SQLite-backed resume state.

The runner wires these concerns together; a matrix run reuses one backend while
executing each configured dataset lane and variant. The same processor is used
for local inference and external batch response parsing.

## 3. Read a configuration

Start with [matrix_smoke.yaml](../experiments/matrix_smoke.yaml), then compare it
with [cpu_llama_cpp_smollm2.yaml](../experiments/cpu_llama_cpp_smollm2.yaml).
Each configuration declares `run`, `model`, `variants`, `input` (or
`datasets`), and `output`. A variant supplies Markdown prompt files plus an
ordered processor pipeline:

```yaml
processor:
  result: single_label
  stages:
    - type: json_decode
    - type: json_schema
      schema: schemas/single-label.json
      enum_from: dataset_labels
```

Run `validate` before an expensive model run. Use repeatable `--set
path.to.value=value` overrides for machine-specific paths, model settings, or
small row limits; overrides are applied before typed validation.

## 4. Locate the implementation

The package uses a dependency direction that makes changes local:

| If you want to change… | Start in… | Then register it in… |
| --- | --- | --- |
| Model transport | `src/llms_experiments/backends/` | `backends/factory.py` |
| Input format | `src/llms_experiments/inputs/` | `inputs/factory.py` |
| Parsing/enrichment strategy | `src/llms_experiments/processors/` | `processors/factory.py` |
| Result serialization | `src/llms_experiments/outputs/` | `outputs/factory.py` |
| YAML validation/defaults | `config.py`, `configuration.py` | generated schema in `docs/` |
| Prompt rendering | `prompting.py` and `experiment-cli/prompt/` | configuration paths |
| Retry, batching, resume, matrix flow | `orchestration.py`, `batching.py` | runtime config |
| Contract rows/statuses | `results.py` and `outputs/schema.py` | result contract docs |

`backend.py`, `input.py`, `prompt.py`, and `_core.py` are compatibility facades
for older imports. New code should import from the plural packages and their
abstract contracts.

## 5. Add a feature safely

Implement the smallest abstract interface possible, add one explicit factory
entry, and add focused tests beside the existing tests:

- backends implement `Backend.generate()` and optionally `close()`;
- readers implement `InputReader.iter_rows()` and preserve deterministic
  positions;
- stages implement `ProcessingStage.requirements()` and `process()` (or
  `prepare_rows()` for row fan-out);
- output stores implement `OutputStore.open_writer()` and `iter_file()`.

Keep processors backend-independent and keep backends unaware of semantic
strategies. Run the full CPU-only checks before committing:

```bash
uv run ruff check .
uv run ruff format --check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run python -m build
```

For the full contracts and strategy details, continue to [Architecture](architecture.md),
[Configuration](configuration.md), [Processing strategies](strategies_review.md),
and [Result contract](result-contract.md).
