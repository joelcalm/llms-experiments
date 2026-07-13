# Generic experiment CLI

`experiment_cli.py` runs local LLM experiments without embedding an application domain in Python. The YAML configuration chooses the input columns, model, prompt fragments, output schemas, candidate sets, batching policy, and result location.

## Layout

```text
experiment-cli/
  experiment_cli.py        # the only Python implementation file
  config/                  # YAML configurations
  data/                    # input fixtures
  prompt/                  # Markdown prompt fragments and JSON schemas
```

Each support directory is one level deep. Generated `outputs/` and `logs/` are ignored by Git.

## Quick start

From the repository root, create the uv environment and run the deterministic local contract check:

```bash
uv sync
uv run python experiment-cli/experiment_cli.py validate \
  --config experiment-cli/config/local_all_modes_smoke.yaml

uv run python experiment-cli/experiment_cli.py self-test \
  --config experiment-cli/config/local_all_modes_smoke.yaml
```

`self-test` uses the built-in fake backend, writes only temporary files, and checks all five request modes plus resume behaviour.

## Run locally on the GPU

```bash
uv run python experiment-cli/experiment_cli.py run \
  --config experiment-cli/config/local_all_modes_smoke.yaml
```

The smoke configuration loads `Qwen/Qwen3.5-0.8B` once and runs, in sequence:

1. single-label JSON;
2. multi-label JSON;
3. ordinal-score JSON;
4. candidate code log-probabilities;
5. independent yes/no log-probabilities.

For JSON variants, the JSON Schema is used both to constrain vLLM output and to validate the saved response. Invalid results are retried according to `validation.retry`; final failures remain in Parquet rather than being dropped.

The runner tunes a batch size separately per variant. The manifest records every candidate, throughput, latency, telemetry snapshot, and selected size. Re-running a completed configuration resumes by `(variant_id, input_row_id)` and does not load a model when no rows remain.

## Artemisa: `vllm run-batch`

For a scheduled Artemisa job, prepare OpenAI Batch-compatible JSONL without loading a model:

```bash
uv run python experiment-cli/experiment_cli.py prepare \
  --config experiment-cli/config/local_all_modes_smoke.yaml

uv run python experiment-cli/experiment_cli.py batch-command \
  --config experiment-cli/config/local_all_modes_smoke.yaml
```

The first command writes `experiment-cli/outputs/local_all_modes_smoke/requests.jsonl`. The second prints the exact command for the worker, equivalent to:

```bash
uv run vllm run-batch \
  -i experiment-cli/outputs/local_all_modes_smoke/requests.jsonl \
  -o experiment-cli/outputs/local_all_modes_smoke/responses.jsonl \
  --model Qwen/Qwen3.5-0.8B \
  --gpu-memory-utilization 0.92 --max-model-len 2048 --max-num-seqs 128 \
  --enable-prefix-caching
```

`requests.jsonl` uses the vLLM/OpenAI Batch form: a unique `custom_id`, `POST`, `/v1/chat/completions`, and a request body. JSON variants carry `response_format: json_schema`; scoring variants request one token with top log-probabilities. Thus all five configured strategies can use `run-batch` in one Artemisa job.

After the worker writes `responses.jsonl`, parse it without GPU inference:

```bash
uv run python experiment-cli/experiment_cli.py parse \
  --config experiment-cli/config/local_all_modes_smoke.yaml \
  --responses experiment-cli/outputs/local_all_modes_smoke/responses.jsonl
```

The parser checks response status and JSON schemas, extracts top token log-probabilities for the two scoring variants, and writes the same Parquet contract and manifest as `run`. It keeps invalid or missing remote results as explicit failures, which makes a follow-up batch safe to prepare instead of silently losing rows.

## Outputs

For `output.directory: outputs/my_run`, the runner writes:

```text
outputs/my_run/
  results.parquet
  single_label_json.parquet
  multi_label_json.parquet
  ordinal_score_json.parquet
  single_label_code_logits.parquet
  independent_yes_no_logits.parquet
  manifest.json
  effective_config.yaml
```

The matching human-readable log and JSONL event log are placed in `logs/` as configured.

## Create a configuration

Copy `config/local_all_modes_smoke.yaml`, then change only data:

- `input`: the source path, format, and ID/text columns;
- `model`: the vLLM model and GPU limits;
- `variants`: prompt files, schemas, or candidate strings;
- `batch`: candidate sizes and warm-up row count;
- `output` and `logging`: ignored local destinations.

Prompt substitutions are intentionally limited to `{{text}}`, `{{row_id}}`, `{{output_schema}}`, `{{raw_response}}`, and `{{validation_errors}}`.
