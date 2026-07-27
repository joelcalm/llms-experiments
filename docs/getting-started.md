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

## 2. Repository experiments

### Why experiments live in the repository

An experiment is a reusable, version-controlled inference recipe. Keeping its
YAML configuration, prompt assets, input provenance, and output contract
together makes a run reviewable and repeatable, rather than an ad-hoc script
or notebook invocation.

### What an experiment is

An experiment reads one or more structured input rows. A row normally contains
an identifier and a text field; it may also carry labels and other source
metadata. The text field contains the sentence that the experiment processes;
the framework preserves its source identity and passes the configured text and
metadata to the prompt renderer.

For each row and configured variant, the LLM executes the configured rendered
prompt and the runner writes a semantic result. The processor declares the
response evidence it needs and interprets it; it can, for example, validate a
JSON label, score candidate tokens, or make one yes/no score per target label.

### Define a concrete experiment

Define a run in YAML and keep prompt templates in external Markdown files. The
complete, runnable [matrix_smoke.yaml](../experiments/matrix_smoke.yaml) is the
recommended example. It defines four dataset lanes, six variants, prompt
assets, processor stages, batching and retry policies, and output locations;
it uses the deterministic `fake` backend so it needs no downloaded model or
API credentials. Validate and run it with:

```bash
uv run llms-experiments validate experiments/matrix_smoke.yaml
uv run llms-experiments run experiments/matrix_smoke.yaml --backend fake --rows 2
```

### Parts of an experiment

1. **YAML configuration** selects and validates the complete run recipe.
2. **Input** provides stable row identifiers, text, and optional source labels.
3. **Processor** prepares rows and interprets raw model evidence.
4. **Backend** transports generation requests without assigning their meaning.
5. **Output** is a Result Contract 2.0 row, with provenance and status.
6. **Writer** durably publishes Parquet, CSV, or JSONL parts and final projections.
7. **Batching and retries** bound resource use, tune or fix batch sizes, and resume safely after interruption.

## 3. Follow one sentence through a run

The following sequence is the whole process. It is also the boundary map for
the repository: each numbered responsibility is deliberately independent of
the others.

```text
experiment YAML + Markdown prompts
  -> configuration validation and effective configuration
  -> input reader normalizes a source sentence
  -> processor prepares a request row and declares requirements
  -> prompt renderer builds the request
  -> backend returns raw model evidence
  -> processor validates/interprets that evidence
  -> output writer durably publishes Result Contract 2.0 rows
```

1. **Process the YAML file.** `configuration.py` loads the experiment,
   resolves environment references and command-line overrides, validates the
   typed configuration, and selects each dataset lane and variant. The
   effective configuration and its hash are recorded in the run manifest.
2. **Read input.** An input reader turns CSV, TSV, JSONL, Parquet, nested JSON,
   or paired TSV into normalized rows. Each sentence retains its input ID and
   receives a deterministic `_source_position`, so it can be traced through a
   resumed run.
3. **Prepare with the processor.** The processor may preserve a row as-is or
   fan it out over labels. It then compiles the ordered processing stages and
   declares one backend-neutral `GenerationRequest`—for example, plain text,
   constrained JSON, or candidate-token log probabilities.
4. **Render prompts and call the backend.** Markdown prompt assets are rendered
   with the normalized row and variant context. The selected backend (vLLM,
   llama.cpp, OpenAI-compatible endpoint, or `fake`) receives the prompt and
   request. Its only responsibility is to return raw text, token evidence,
   timing, and transport errors.
5. **Interpret the response.** The processor converts the raw response into a
   semantic result. Its stages can decode JSON, validate a schema, aggregate
   candidate scores, or enrich confidence. This is why backends never parse
   labels and processors never depend on a particular model transport.
6. **Write output.** The output store writes Result Contract 2.0 rows to
   Parquet, CSV, or JSONL. It first publishes append-only parts atomically,
   then commits resume identities to SQLite and produces combined and
   per-variant result files. The manifest records provenance, configuration,
   model, status, and resource diagnostics.
7. **Batch, retry, and resume.** Orchestration bounds memory with streaming,
   runs fixed or adaptive batches, retries configured transient/validation
   cases, and resumes only from matching durable rows. Matrix runs repeat the
   same flow for every dataset lane and variant while reusing a compatible
   backend.

External batch preparation and parsing use the same request, raw-response,
processor, and output boundaries. Only the transport timing changes.

## 4. Read a configuration

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

## 5. Find the repository parts

The package uses a dependency direction that makes changes local. Use this map
after following the run above to find the implementation responsible for each
part:

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

## 6. Add a feature safely

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
