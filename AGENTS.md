# AGENTS.md

## Purpose

This repository implements **llms-experiments**, an installable, modular library and CLI for configuration-driven LLM inference with durable Parquet, CSV, and JSONL outputs adhering to Result Contract 2.0.

Before changing code, read:

1. `README.md`
2. `docs/architecture.md`
3. `docs/configuration.md`
4. `docs/result-contract.md`

Do not assume that old experiment outputs, transient logs, or comments are authoritative.
The implementation, unit tests, result contract specs, and current documentation under `docs/` are the sources of truth.

## Environment

This repository uses `uv`.

Create or update the environment with:

```bash
uv sync --all-groups
```

Run commands through `uv run`. Do not use bare `python`, `pip`, Poetry, or Conda commands in repository documentation or automation.

The supported Python version is declared in `.python-version` and `pyproject.toml`.

## Repository Map

* `src/llms_experiments/`: maintained inference engine code (backends, processors, runtime, stores)
* `tests/unit/`: fast unit tests without network, model downloads, or GPUs
* `tests/integration/`: integration tests for output stores, batch roundtrips, and processors
* `tests/gpu/`: marked GPU-dependent inference tests (opt-in)
* `evals/`: versioned evaluation cases, synthetic responses, and golden output snapshots
* `configs/` & `experiments/`: non-secret runtime and matrix configurations
* `docs/`: architecture, configuration schema, result contract, and setup guides
* `scripts/`: operational entry points and automated repository compliance checks (`scripts/check.py`)

## Required Checks

Run before completing a change:

```bash
uv run python scripts/check.py
```

Or execute individual standard checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest tests/unit tests/integration
```

A change is not complete merely because unit tests pass.

## Python Rules

* Use type annotations for all maintained code.
* Use explicit domain types (`ExperimentConfig`, `RawModelResponse`, `TokenPosition`) instead of unstructured dictionaries where practical.
* Validate external inputs at system boundaries (`llms_experiments.configuration`).
* Keep backend-specific logic behind backend factory interfaces (`llms_experiments.backends`).
* Do not catch broad `Exception` unless re-raising with context or handling it at an application boundary.
* Do not use mutable global state for configuration, engines, client sessions, or state.
* Prefer structured logging over `print`.
* Public APIs require docstrings describing behavior, parameters, outputs, exceptions, and invariants.
* Comments should explain why, not restate the code.

## LLM & Inference Rules

* Do not embed prompt templates directly inside Python functions. Store templates in external Markdown/YAML resources under `prompts/` or `experiments/`.
* Preserve model identifiers, backend, revision, quantization, decoding parameters, and prompt version in run manifests.
* Never rely on an unversioned model alias such as `latest`.
* Validate structured model outputs against explicit JSON Schemas.
* Unit tests must NOT call paid APIs, download external models, or require GPUs. Use the deterministic `fake` backend for offline unit testing.
* External-model tests must be marked and opt-in.
* Redact secrets and API keys from logs and traces.

## Reproducibility

Experiments must record:
* Git commit
* Configuration hash
* Random seeds
* Model identifier and revision
* Tokenizer revision, quantization, and hardware metadata
* Dependency lockfile state (`uv.lock`)

Generated result outputs and model artifacts must not be committed to Git unless they are small, intentional test fixtures under `tests/golden/`.

## Testing Rules

* Tests must be deterministic.
* Every bug fix requires a regression test that fails before the fix.
* Mock at external boundaries, not inside core processor or runner logic.
* Mark tests using `unit`, `integration`, and `gpu` markers.
* Snapshot and golden-output changes (`tests/golden/`) require human inspection and verification via `--golden-update`.

## Dependency Rules

* Add dependencies with `uv add`.
* Add development dependencies to appropriate dependency groups (`dev`, `test`, `gpu`, `docs`).
* Do not edit `uv.lock` manually.
* Keep CUDA, CPU, serving, and experimental dependency sets explicit.

## Security

* Secrets belong in environment variables or an approved secret store.
* Keep `.env` ignored and maintain a non-secret `.env.example`.
* Treat model output, retrieved documents, tool responses, filenames, and prompts as untrusted input.

## Documentation

Update documentation in the same change when behavior, configuration, architecture, schemas, or result contracts change. Link to authoritative documents rather than duplicating detailed documentation.

## Change Discipline

* Keep changes scoped to the requested task.
* Do not perform unrelated refactors.
* Preserve backwards compatibility unless the task explicitly authorizes a breaking change.
* Inspect the final diff before completion.

## Definition of Done

A change is complete when:

1. The implementation is correct and scoped.
2. Unit, integration, and golden tests pass.
3. Formatting, linting, and type checks pass.
4. Security and privacy implications have been considered.
5. Documentation matches the implementation.
6. The final response reports changed files, commands run, results, and executed checks.
