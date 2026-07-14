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

## Open blockers before full evaluation

1. `experiments/ministral_all_datasets.yaml` is currently an interface smoke
   configuration. Its `alpha/beta/gamma`, `A/B/C`, and `yes/no` choices do not
   represent the MFTC, MFRC, ValueEval, or ProtoEthos label vocabularies. A
   full run with this file is not a meaningful accuracy evaluation.
2. The five-dataset matrix eagerly materialises the 2.55M-row ProtoEthos lanes
   and all variant prompts. Artemisa needs streaming/chunked input and
   append-only Parquet parts rather than whole-run in-memory lists and rewrites.
3. Prompt assets place `{{text}}` before the JSON schema in some variants and
   no prompt-family scheduler exists yet. Static prompt fragments should be
   grouped before row text, then scheduled by prefix/mode/schema.
4. External `parse` validates remote responses once; configured correction
   retries and complete remote usage/latency metadata are not yet reproduced.
5. Resume checks source position but does not fail when model, prompt assets,
   source data, or generation configuration changes in an existing output.
6. The current ValueEval materialised source contains 5,393 rows, while the
   raw train/validation/test files contain 8,865 rows in total.
7. The Ministral YAML uses host-specific absolute data paths. Artemisa needs
   portable paths or per-worker path overrides.

Verification after the fixes:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check experiment-cli/experiment_cli.py
UV_CACHE_DIR=.uv-cache uv run ruff format --check experiment-cli/experiment_cli.py
UV_CACHE_DIR=.uv-cache uv run python experiment-cli/experiment_cli.py self-test \
  --config experiments/local_all_modes_smoke.yaml --backend fake
```
