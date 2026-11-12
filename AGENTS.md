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

(Today there is only one dependency group, `dev`, so this is equivalent to `uv sync`; prefer `--all-groups` so the command keeps working if more groups are added later.)

Run commands through `uv run`. Do not use bare `python`, `pip`, Poetry, or Conda commands in repository documentation or automation.

The supported Python version is declared in `.python-version` and `pyproject.toml`.

## Repository Map

* `src/llms_experiments/`: maintained inference engine code (backends, processors, runtime, stores)
* `tests/`: flat test directory containing both `llms_experiments` package tests and the separate `experiment-cli` subprocess-based test suite (see `tests/conftest.py`). The `unit`, `integration`, and `gpu` pytest markers are registered in `pyproject.toml` but tests are not yet split into subdirectories or fully marked; use markers where practical when adding or modifying tests.
* `evals/`: versioned evaluation cases, synthetic responses, and golden output snapshots
* `experiments/`: non-secret runtime and matrix configurations
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
uv run pytest tests
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

### Git hygiene and large files

* **Never commit large derived artifacts.** Anything over roughly **100 MB**
  (parsed result Parquet under `release/`, model checkpoints, full `results/`
  trees) is a published artifact, not repository source. Upload it to external
  or release storage (GitHub Releases, an artifact store, Lustre) and keep its
  path in `.gitignore`. `results/`, `release/`, `outputs/`, `logs/`, and
  `internal_documentation/` are already ignored for this reason.
* **Per-experiment job scripts are runtime artifacts.** HTCondor `*.sub`
  descriptors and Condor/Slurm worker shells are generated per campaign; keep
  only tracked, reusable templates (`condor/submit.htcondor`, `condor/run.sh`,
  `slurm/submit-vllm.sh`, `examples/run_local_gpu.sh`) and git-ignore the
  generated ones. Do not commit a matrix of one-off submit files.
* **Before committing, scan for bloat.** Run
  `git status --porcelain | du -sh` style checks and `du -sh` on untracked
  directories so a multi-hundred-MB directory is never staged by accident.
* Treat `docs/` (architecture, configuration, result-contract) as the
  human-readable contract, but the implementation, unit tests, and
  `result-contract.md` remain the sources of truth when they disagree.

## Testing Rules

* Tests must be deterministic.
* Every bug fix requires a regression test that fails before the fix.
* Mock at external boundaries, not inside core processor or runner logic.
* The `unit`, `integration`, and `gpu` pytest markers are registered in `pyproject.toml`; use them to mark new or modified tests where practical, even though the existing suite is not fully marked yet.
* `tests/` is flat and holds two suites together: `llms_experiments` package tests, and `experiment-cli` subprocess-based tests (see the scope comment at the top of `tests/conftest.py`). Keep this in mind before assuming a fixture or helper applies repo-wide.
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

Documentation is **not** authoritative on its own: when a doc disagrees with the
implementation, unit tests, or `docs/result-contract.md`, the code is correct and
the doc must be fixed. Prefer targeted edits to the existing doc over creating
new files.

### Audit workflow

For a broad documentation/structure/efficiency audit, parallelize the independent
checks with subagents (one each for doc-staleness, structure, efficiency) rather
than reading the whole tree serially, then reconcile findings. Always pair an audit
with `uv run python scripts/check.py` and a `git status`/`du -sh` bloat scan before
declaring the tree clean.

### Current Artemisa operations

Before inspecting or changing the ProtoEthos INT4 campaign, read the concise
live recovery record in
[`internal_documentation/INT4_RECOVERY_STATUS_20260831.md`](internal_documentation/INT4_RECOVERY_STATUS_20260831.md).
It defines the authoritative completion check, durable result roots, hardware
split, duplicate policy, and current resumable lanes.

For future campaigns: inspect Condor and per-lane checkpoints first; reuse
matching completed or partial trees; never queue an identical model revision,
dataset, and JSON strategy twice; keep dense large models on H100 and MoE or
small models on A100; keep durable results outside the known quota-limited
project root; use `/tmp` only for rebuildable compiler/runtime cache and
transient logs; and record the post-action queue snapshot before the next
change. Parts alone are progress, not completion.

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
