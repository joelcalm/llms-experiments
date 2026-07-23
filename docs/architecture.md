# Architecture

The package is split by responsibility:

- `config`: typed YAML contracts, dotted overrides, and dataset selection;
- `input`: bounded CSV, JSON, JSONL, TSV, and Parquet readers;
- `prompt`: cached Markdown/schema assets and reusable static prefixes;
- `backend`: vLLM, OpenAI-compatible endpoint, and fake implementations;
- `runner`: persistent-engine single and matrix orchestration;
- `persistence`: atomic Parquet parts and SQLite resume state;
- `external_batch`: `vllm run-batch` preparation and parsing;
- `cli`: the five supported user commands.

The internal `_core` module retains shared implementation while the v0.2 module
boundaries stabilise. It is not public API. Evaluation logic is deliberately
absent from every layer.

## Run flow

The loader validates one YAML contract and resolves prompt/schema assets. A run
creates one backend, then processes variants in order. Each variant reuses its
static prompt prefix, tunes a bounded batch size, streams inputs, validates
responses, and writes append-only parts. Deferred retries use smaller batches.
The same backend remains alive across a matrix.

`PartWriter` writes a temporary Parquet file and atomically renames it. Only
after publication are row keys committed to the SQLite resume index. On restart,
the index is seeded only from durable parts whose configuration hashes match.
Final projections are merged from parts without changing their identities.

External batch preparation emits OpenAI Batch-shaped JSONL. Parsing uses the
same response validation and result-row builder as in-process inference, so all
backends converge on contract 2.0.
