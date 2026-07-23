# Contributing to llms-experiments

Contributions should keep the runner domain-agnostic and preserve the Parquet result contract. Evaluation metrics belong outside this repository.

## Development Setup

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
python -m build
```

Use the fake backend and standard test fixtures for unit tests. Tests must not require private datasets, model downloads, or a GPU. If an intentional change affects result rows or manifests, regenerate golden files with `uv run pytest --golden-update` and review the resulting diff.

Keep changes focused, document user-visible configuration or schema changes, and avoid machine-specific paths in committed files.

Do not add ProtoEthos or another evaluation package as a dependency. Result contract changes require golden fixtures, schema documentation, and a changelog entry. GPU changes additionally require the separate model/throughput verification gate.
