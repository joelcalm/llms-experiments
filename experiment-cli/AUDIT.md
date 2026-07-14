# CLI audit

Audited 14 July 2026 with three read-only subagents covering prompt/cache
construction, configuration/data/resume behaviour, and vLLM/API/run-batch
execution. The audit findings below are deliberately separated into fixes
already applied and issues that must be resolved before a full evaluation run.

## Fixed in this pass

- Direct Python inference now uses `LLM.chat`, matching the chat-template path
  used by API and `run-batch`.
- Candidate instructions and candidate values are present in the rendered
  prompt; YAML `yes`/`no` values are quoted.
- Batch parsing preserves source delimiters, filters, labels, dataset IDs, and
  source-position-based IDs. Duplicate response IDs are rejected.
- Duplicate source IDs no longer collide in resume keys or `custom_id` values.
- Result rows include serialised `gold_labels` when the source provides them.
- Runtime response-count checks, OOM batch backoff, safer engine shutdown,
  unknown-placeholder errors, per-batch GPU snapshots, and positive row-limit
  checks were added.
- Matrix lane selection, matrix preparation, and overridden output log paths
  are isolated for parallel jobs.
- Dataset-specific label/code vocabularies and MFT/SHVT theory fragments are
  declared in YAML; the runner remains generic.
- Large CSV inputs stream through bounded batches into append-only Parquet
  parts, with a disk-backed resume index and final merge.
- Static prompt parts are assembled before the final `input.md` fragment and
  each variant records a reusable `prompt_group_id`.
- External parsing emits bounded retry JSONL requests and consumes retry
  responses, preserving attempt counts and final validation status.
- Resume keys include effective model/input/prompt/config hashes and manifests
  record source-file provenance metadata.

## Remaining operational checks

1. The real Ministral GPU acceptance run requires an environment with an
   accessible NVIDIA device and compatible driver.
2. ValueEval and ProtoEthos label strings are intentionally preserved exactly
   as provided by their source files. Downstream scoring should use the same
   canonicalisation when comparing predictions and gold labels.
3. `source_provenance.metadata_hash` hashes file metadata (size and mtime),
   not the full bytes; enable an external content checksum when the data
   publication requires byte-level provenance.

Verification after the fixes:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check experiment-cli/experiment_cli.py
UV_CACHE_DIR=.uv-cache uv run ruff format --check experiment-cli/experiment_cli.py
UV_CACHE_DIR=.uv-cache uv run python experiment-cli/experiment_cli.py self-test \
  --config experiments/local_all_modes_smoke.yaml
```
