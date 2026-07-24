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
- [Configuration JSON Schema](config.schema.json): generated editor/tooling
  schema. Regenerate it with `python scripts/generate_config_schema.py` from an
  installed checkout.

The source tree also contains runnable examples under `experiments/`. The
CPU-only fake-backend matrix is the fastest installation check; the llama.cpp
example requires the optional `llama-cpp` dependency and a local GGUF model.
