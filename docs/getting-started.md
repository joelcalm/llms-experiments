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

### What an experiment does

An experiment reads one or more structured input rows. A row normally contains
an identifier and a text field; it may also carry labels and other source
metadata. For a corpus of action-associated sentences, the text field can hold
a sentence such as $CE_{X:a}$. The framework does not assign a special meaning
to that notation: it preserves the source identity and passes the configured
text and metadata to the prompt renderer.

For each row and configured variant, the runner renders the prompt, asks the
selected LLM backend for the evidence required by the processor, and writes a
semantic result. A processor can, for example, validate a JSON label, score
candidate tokens, or make one yes/no score per target label.

### Define a concrete experiment

Define a run in YAML and keep prompt templates in external Markdown files. The
following complete, runnable example is available as
[matrix_smoke.yaml](../experiments/matrix_smoke.yaml). It uses the deterministic
`fake` backend and exercises multiple input formats and processor strategies:

```yaml
run:
  id: example_single_label
  seed: 123

input:
  path: data/actions.jsonl
  format: jsonl
  id_column: id
  text_column: sentence
  labels: [care, harm]

model:
  name: fake
  backend: fake

variants:
  - id: classify_action_sentence
    max_tokens: 32
    prompts:
      - prompts/system.md
      - prompts/classify-action.md
      - prompts/input.md
    processor:
      result: single_label
      stages:
        - type: json_decode
        - type: json_schema
          schema: schemas/single-label.json
          enum_from: dataset_labels

output:
  directory: results/example_single_label
  format: parquet
```

The referenced prompt files and JSON schema are experiment assets: create them
for a new configuration, or start from the paths used by the checked-in smoke
example. Validate before running:

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

## 3. Understand one run

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

## 5. Sentence generation: proposed extension

Sentence generation from semantic frames is a useful adjacent workflow, but it
is not implemented by this inference-runner today. In particular, the current
configuration schema has no `recipe`, semantic-frame, linguistic-control,
realiser, or transformation sections. It would be misleading to present it as
a runnable experiment configuration until those contracts and implementations
exist.

The intended design can fit the repository's existing boundaries:

1. A versioned YAML **recipe** would define a set of semantic frames and a set
   of linguistic controls. Together, one frame and one control form a sample:
   a realisation recipe rather than a generated sentence.
2. A generation **engine** would iterate frames, produce *m* controlled
   realisations per frame, bind roles, and produce *n* template-based
   instantiations initially.
3. A pluggable **realiser** would turn each instantiation into text. A
   template realiser is an appropriate first implementation; an inflation- or
   grammar-based realiser could use the same interface later.
4. Ordered **transformations** would receive `(semantic_frame,
   linguistic_control, sentence)` and return a transformed sample. An abstract
   transformation interface would keep these operations composable and
   testable.
5. An **output writer** would persist the source frame, controls,
   instantiation provenance, transformations, and generated sentence. This is
   analogous to the inference result writer, but it needs its own versioned
   generation-data contract rather than Result Contract 2.0, which covers LLM
   inference results only.

This is a sensible repository feature, but it should be built as a separate
sentence-generation subsystem—not squeezed into the existing `input` reader
or processor pipeline. That separation preserves the current guarantee that a
processor interprets model evidence for an already supplied input row. When
implemented, its recipe schema, sample identity rules, deterministic seeding,
and output contract should be documented and tested before this guide shows a
runnable YAML example.

## 6. Locate the implementation

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

## 7. Add a feature safely

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
