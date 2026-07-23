# llms-experiments

[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Documentation](https://img.shields.io/badge/docs-v1.0.0-blue.svg)](docs/)

`llms-experiments` is an installable CLI for configuration-driven Large Language Model (LLM) inference with durable, resumable Apache Parquet outputs. It supports an in-process vLLM inference engine, vendor-neutral OpenAI-compatible endpoints, external `vllm run-batch` preparation, and a deterministic fake backend for offline testing.

This package executes inference runs and records durable predictions. It does not calculate accuracy, F1, diagnostic axioms, or aggregate evaluation reports. Its interoperability boundary is the versioned result contract 2.0.

---

## Key Features

- **Configuration-Driven Execution:** Runs are fully specified by YAML configurations and Markdown prompt templates.
- **Engine Persistence:** Model engines remain warm across matrix evaluations to eliminate reload overheads.
- **Resumable & Atomic Operations:** State is tracked in a local SQLite database. Outputs are written atomically to prevent corrupted Parquet files during unexpected shutdowns.
- **Adaptive Batch Tuning:** Automatically tunes GPU batch sizes dynamically,halving sizes upon out-of-memory events and retrying safely.
- **Multiple Backend Support:** Seamlessly switches between local vLLM, vendor-neutral OpenAI API endpoints, and fake test backends.

---

## Installation

Python 3.11–3.13 is supported. The default installation is CPU-light and validates configurations, queries endpoints, prepares/parses external batches, and executes tests:

```bash
python -m pip install .
```

To install local vLLM GPU inference capabilities:

```bash
python -m pip install '.[gpu]'
```

The GPU extra uses `vllm>=0.25.1,<0.26`, including native Gemma 4 Unified and Qwen support. The repository carries no custom model monkeypatches or startup hooks.

---

## Commands & Usage

```bash
llms-experiments validate CONFIG
llms-experiments run CONFIG
llms-experiments prepare CONFIG
llms-experiments parse CONFIG --responses RESPONSES.jsonl
llms-experiments doctor
```

- `validate`: Enforces typed Pydantic configuration validation and verifies prompt template paths.
- `run`: Automatically processes a single dataset or a configured dataset matrix. Supports repeatable overrides such as `--set batch.size=16`, as well as `--datasets` and `--variants` flags.
- `prepare`: Formats batch request files and prints matching `vllm run-batch` launch commands.
- `parse`: Converts external batch responses into contract-compliant result Parquet files.
- `doctor`: Reports driver, CUDA, and package environment diagnostics.

---

## Configuration

Each run is specified in a single YAML file. Inputs, models, Markdown prompt fragments, structured output schemas, CPU resource limits, retries, batching policies, and output paths remain data, not Python code. Validate the included CPU-only matrix fixture with:

```bash
llms-experiments validate experiments/matrix_smoke.yaml
llms-experiments run experiments/matrix_smoke.yaml --backend fake --rows 4
```

The typed configuration models reside in `llms_experiments.config`; the generated [JSON Schema](docs/config.schema.json) supports editors and external tooling. See [configuration.md](docs/configuration.md) for the complete contract.

---

## Efficiency and Durability

A single model engine is reused for an entire run matrix. Static prompt prefixes and schema assets are cached, prefix caching is enabled by default, and batch size is tuned per variant with smaller deferred retries. Streaming readers bound host memory usage. CPU affinity and native thread pools are YAML-controlled.

Rows are buffered into append-only Parquet parts. A part is atomically renamed into place before its keys are committed to SQLite resume state, ensuring interrupted buffers never become corrupt. Failed rows remain explicit, and backend outages are retried on resume without duplicating result identities.

See [architecture.md](docs/architecture.md) and the independent [result contract](docs/result-contract.md).

---

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
python -m build
```

The test suite is CPU-only and uses the fake backend. GPU throughput and model acceptance remain separate release gates.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [MIT Licence](LICENSE).
