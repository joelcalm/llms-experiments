# Documentation

These pages describe the installable `llms-experiments` package at version
`0.2.x` and its five CLI commands (`validate`, `run`, `prepare`, `parse`, and
`doctor`).

- [Getting started](getting-started.md): installation, the end-to-end data
  flow, code navigation, and how to add an extension.
- [Architecture](architecture.md): package boundaries, processor compilation,
  extension points, and durable resume behavior.
- [Configuration](configuration.md): the YAML contract, supported inputs,
  backends, processor stages, and output formats.
- [Processing strategies](strategies_review.md): concrete YAML pipelines for
  structured labels, ordinal scores, candidate log probabilities, fan-out
  scoring, and verbalized confidence.
- [Result contract](result-contract.md): the row schema, status semantics,
  provenance, and output files.
- [ProtoEthos 18-model results](protoethos_18_model_results.md): result paths,
  JSON strategy prompts, checkpoint precision, and current file-location audit.
- [Configuration JSON Schema](config.schema.json): generated editor/tooling
  schema. Regenerate it with `uv run python scripts/generate_config_schema.py`
  from an installed checkout.

The source tree also contains runnable examples under `experiments/`. The
CPU-only fake-backend matrix is the fastest installation check; the llama.cpp
example requires the optional `llama-cpp` dependency and a local GGUF model.

Artemisa execution reports are kept separately under
[`../internal_documentation/`](../internal_documentation/). For the current
ProtoEthos INT4 JSON-only campaign, read
[`00-live-status.md`](../internal_documentation/00-live-status.md) and
[`INT4_RECOVERY_STATUS_20260831.md`](../internal_documentation/INT4_RECOVERY_STATUS_20260831.md);
dated status files are provenance snapshots. Re-query the scheduler before
treating a job or result as current.
