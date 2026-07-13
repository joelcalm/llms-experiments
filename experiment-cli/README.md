# experiment-cli

`experiment-cli` is a configuration-driven runner for local LLM experiments. It deliberately contains no theory, classifier, or application-specific assumptions: variants, prompt fragments, schemas, candidate sets, output paths, and batching policy are YAML data.

Run the local configuration checks with:

```bash
cd protoethos
uv run python llms-experiments/experiment-cli/cli.py validate --config llms-experiments/experiment-cli/configs/local_all_modes_smoke.yaml
```

Generate the 128-row Parquet smoke input once if it is absent:

```bash
uv run python llms-experiments/experiment-cli/data/make_smoke_data.py
```

The configured `run` command loads one vLLM instance and processes all variants sequentially. It tunes each variant independently, records rejected candidates and GPU snapshots (including PyTorch allocated/reserved peaks where available), retries failed structured outputs when enabled, and resumes by `(variant_id, input_row_id)` without duplicate rows. Set `model.synchronize_cuda: true` when measuring fully synchronised timings.

```bash
uv run python llms-experiments/experiment-cli/cli.py run --config llms-experiments/experiment-cli/configs/local_all_modes_smoke.yaml
```

`prepare` writes request JSONL, while `parse --responses PATH` consumes a `responses.jsonl` whose rows have `variant_id`, `input_row_id`, and `response` (or `raw_response`). Both use the same output and validation contracts.

The actual smoke configuration requires vLLM and a functioning NVIDIA driver. `run` reports a clear preflight error if `nvidia-smi` is unavailable or cannot communicate with the driver.
