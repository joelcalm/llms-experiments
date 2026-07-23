# llms-experiments

`llms-experiments` is an installable CLI for configuration-driven LLM
inference with durable, resumable Parquet outputs. It supports an in-process
vLLM engine, vendor-neutral OpenAI-compatible endpoints, external
`vllm run-batch`, and a deterministic fake backend for tests.

This project does not calculate accuracy, F1, diagnostic axioms, or aggregate
evaluation reports. Its interoperability boundary is the documented result
contract 2.0.

## Install

Python 3.11–3.13 is supported. The default installation is CPU-light and can
validate configs, call endpoints, prepare/parse external batches, and run tests:

```bash
python -m pip install .
```

Install local vLLM inference separately:

```bash
python -m pip install '.[gpu]'
```

The GPU extra uses `vllm>=0.25.1,<0.26`, including native Gemma 4 Unified and
Qwen support. The repository carries no Transformers fork, model monkeypatch,
or startup hook.

## Commands

```bash
llms-experiments validate CONFIG
llms-experiments run CONFIG
llms-experiments prepare CONFIG
llms-experiments parse CONFIG --responses RESPONSES.jsonl
llms-experiments doctor
```

`run` automatically handles one input or a configured dataset matrix. Use
repeatable dotted overrides such as `--set batch.size=16`, plus `--datasets`
and `--variants` to select matrix lanes. `prepare` writes batch request files
and prints their matching `vllm run-batch` launch commands.

The legacy `experiment-cli/experiment_cli.py` path forwards to these commands
with a deprecation warning in v0.2. It maps `run-matrix` and `prepare-matrix`
and will be removed in v0.3.

## Configuration

Each run is one YAML file. Inputs, models, Markdown prompt fragments, structured
output schemas, CPU limits, retries, batching, and output paths remain data,
not Python code. Validate the included CPU-only matrix fixture with:

```bash
llms-experiments validate experiments/matrix_smoke.yaml
llms-experiments run experiments/matrix_smoke.yaml --backend fake --rows 4
```

The typed configuration models are in `llms_experiments.config`; the generated
[JSON Schema](docs/config.schema.json) supports editors and external tooling.
See [configuration.md](docs/configuration.md) for the complete contract.

## Efficiency and durability

One model engine is reused for an entire run. Static prompt prefixes and schema
assets are cached, prefix caching is enabled by default, and batch size is tuned
per variant with smaller deferred retries. Streaming readers bound host memory.
CPU affinity and native thread pools are YAML-controlled.

Rows are buffered into append-only Parquet parts. A part is atomically renamed
into place before its keys are committed to SQLite resume state, so interrupted
buffers never become resumable. Failed rows remain explicit and backend outages
are retried on resume without duplicating result identities.

See [architecture.md](docs/architecture.md) and the independent
[result contract](docs/result-contract.md).

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
python -m build
bash -n condor/run.sh
bash -n slurm/submit-vllm.sh
```

The test suite is CPU-only and uses the fake backend. GPU throughput and model
acceptance remain separate release gates; do not claim maximised throughput
until the documented warm-run comparison passes on a working GPU host.

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[MIT License](LICENSE).
