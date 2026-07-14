# Generic experiment CLI

`experiment_cli.py` runs local LLM experiments without embedding an application domain in Python. The YAML configuration chooses the input columns, model, prompt fragments, output schemas, candidate sets, batching policy, and result location.

## Layout

```text
experiment-cli/
  experiment_cli.py        # the only Python implementation file
  ../experiments/          # YAML experiment configurations
  data/                    # input fixtures
  prompt/                  # Markdown prompt fragments and JSON schemas
```

Each support directory is one level deep. Generated `results/` are ignored by Git.

## Quick start

From the repository root, create the uv environment and run the deterministic local contract check:

```bash
uv sync
uv run python experiment-cli/experiment_cli.py validate \
  --config experiments/local_all_modes_smoke.yaml

uv run python experiment-cli/experiment_cli.py self-test \
  --config experiments/local_all_modes_smoke.yaml
```

`self-test` uses the built-in fake backend, writes only temporary files, and checks all five request modes plus resume behaviour.

## Run locally on the GPU

```bash
uv run python experiment-cli/experiment_cli.py run \
  --config experiments/local_all_modes_smoke.yaml
```

The smoke configuration loads `Qwen/Qwen3.5-0.8B` once and runs, in sequence:

1. single-label JSON;
2. multi-label JSON;
3. ordinal-score JSON;
4. candidate code log-probabilities;
5. independent yes/no log-probabilities.

For JSON variants, the JSON Schema is used both to constrain vLLM output and to validate the saved response. Invalid results are retried according to `validation.retry`; final failures remain in Parquet rather than being dropped.

The runner tunes a batch size separately per variant. The manifest records every candidate, throughput, latency, telemetry snapshot, and selected size. Re-running a completed configuration resumes by `(variant_id, input_row_id)` and does not load a model when no rows remain.

## Compare Python, API, and `run-batch`

The same experiment can be benchmarked through three vLLM interfaces:

1. `python`: direct in-process Python API using `LLM.generate`, as in the vLLM example. The model is loaded by `experiment_cli.py` and reused across all variants.
2. `api`: OpenAI-compatible HTTP API. Start a persistent server first, for example:

   ```bash
   uv run vllm serve Qwen/Qwen3.5-0.8B \
     --host 127.0.0.1 --port 8000 \
     --gpu-memory-utilization 0.92 --max-model-len 2048 \
     --max-num-seqs 128 --enable-prefix-caching --language-model-only
   ```

3. `run-batch`: the external vLLM batch CLI. Each repetition starts a `vllm run-batch` process and therefore includes model startup time.

Run each approach separately on the configured 64 rows so that two model
instances never compete for the same GPU. With the API server running, measure
the HTTP path first:

```bash
uv run python experiment-cli/experiment_cli.py benchmark \
  --config experiments/local_all_modes_smoke.yaml \
  --approaches api
```

Stop the server, then measure the other two paths:

```bash
uv run python experiment-cli/experiment_cli.py benchmark \
  --config experiments/local_all_modes_smoke.yaml \
  --approaches python

uv run python experiment-cli/experiment_cli.py benchmark \
  --config experiments/local_all_modes_smoke.yaml \
  --approaches run-batch
```

The report is written to `results/local_all_modes_smoke/benchmark.json` and merges measurements from repeated `--approaches` invocations when the workload is unchanged. It records wall time, requests per second, tokens per second, errors, GPU snapshots, and whether model startup was included. The Python and HTTP measurements assume a warm model; `run-batch` is end-to-end and includes process/model startup. Use `--rows` to change the sample and `--approaches python,api` to run only a subset when the GPU can accommodate the selected services.

The three measurements are intentionally reported separately rather than reduced to one ranking: HTTP includes request/network overhead and depends on server concurrency, while `run-batch` includes a fresh model load. For steady-state inference compare the Python and API measurements; for a scheduled job compare the end-to-end `run-batch` measurement.

## Artemisa: `vllm run-batch`

For a scheduled Artemisa job, prepare OpenAI Batch-compatible JSONL without loading a model:

```bash
uv run python experiment-cli/experiment_cli.py prepare \
  --config experiments/local_all_modes_smoke.yaml

uv run python experiment-cli/experiment_cli.py batch-command \
  --config experiments/local_all_modes_smoke.yaml
```

The first command writes `results/local_all_modes_smoke/requests.jsonl`. The second prints the exact command for the worker, equivalent to:

```bash
uv run vllm run-batch \
  -i results/local_all_modes_smoke/requests.jsonl \
  -o results/local_all_modes_smoke/responses.jsonl \
  --model Qwen/Qwen3.5-0.8B \
  --gpu-memory-utilization 0.92 --max-model-len 2048 --max-num-seqs 128 \
  --enable-prefix-caching
```

`requests.jsonl` uses the vLLM/OpenAI Batch form: a unique `custom_id`, `POST`, `/v1/chat/completions`, and a request body. JSON variants carry `response_format: json_schema`; scoring variants request one token with top log-probabilities. Thus all five configured strategies can use `run-batch` in one Artemisa job.

After the worker writes `responses.jsonl`, parse it without GPU inference:

```bash
uv run python experiment-cli/experiment_cli.py parse \
  --config experiments/local_all_modes_smoke.yaml \
  --responses results/local_all_modes_smoke/responses.jsonl
```

The parser checks response status and JSON schemas, extracts top token log-probabilities for the two scoring variants, and writes the same Parquet contract and manifest as `run`. It keeps invalid or missing remote results as explicit failures, which makes a follow-up batch safe to prepare instead of silently losing rows.

For the five-dataset matrix, prepare independent request files so Artemisa
workers can run them in parallel:

```bash
uv run python experiment-cli/experiment_cli.py prepare-matrix \
  --config experiments/ministral_all_datasets.yaml \
  --datasets mftc,mfrc,valueeval
```

This writes one `results/ministral_all_datasets/dataset=<id>/requests.jsonl`
per selected dataset. Run `batch-command` (or `vllm run-batch` directly) in
each worker with that file and parse each response against the corresponding
dataset configuration. The CLI can print the worker command for a matrix lane:

```bash
uv run python experiment-cli/experiment_cli.py batch-command \
  --config experiments/ministral_all_datasets.yaml --dataset mftc
```

## Outputs

For `output.directory: results/my_run`, the runner writes:

```text
results/my_run/
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

## Launch variants from one YAML

Every command accepts repeatable dotted-path overrides. Values are parsed as
YAML, so strings, numbers, booleans, lists, and nested mappings are supported:

```bash
uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml \
  --run-id ministral_mftc_job_01 \
  --output results/ministral_mftc_job_01 \
  --set datasets.0.input.limit=1000 \
  --set model.max_num_seqs=32
```

Shortcuts are available for `run.id`, `model.name`, `model.backend`, and
`output.directory`; arbitrary fields use `--set KEY=VALUE`. This makes
parallel launches independent: give each process a unique `--run-id` and
`--output`, and optionally restrict its dataset lane with a dotted override.
For example, two workers can target different ProtoEthos slices:

```bash
uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml \
  --datasets protoethos_mft \
  --run-id proto_mft --output results/proto_mft

uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml \
  --datasets protoethos_shvt \
  --run-id proto_shvt --output results/proto_shvt
```

## Create a configuration

Copy `experiments/local_all_modes_smoke.yaml`, then change only data:

- `input`: the source path, format, and ID/text columns;
- `model`: the vLLM model and GPU limits;
- `variants`: prompt files, schemas, or candidate strings;
- `batch`: candidate sizes and warm-up row count;
- `output` and `logging`: ignored local destinations.

Prompt substitutions are intentionally limited to `{{text}}`, `{{row_id}}`,
`{{dataset_id}}`, `{{candidates}}`, `{{output_schema}}`, `{{raw_response}}`,
and `{{validation_errors}}`.

## Five-dataset Ministral matrix

`experiments/ministral_all_datasets.yaml` is the Artemisa-ready matrix for the five
datasets used in this project: MFTC, MFRC, ValueEval, and the MFT and SHVT
slices of ProtoEthos. It uses the model
`mistralai/Ministral-3-3B-Instruct-2512` and the same five interface variants
listed above. Dataset readers are format-driven: nested MFTC JSON, materialised
JSONL records, and the semicolon-delimited ProtoEthos CSV are all normalised to
the same `id`/`text` row contract. Gold labels are retained as `_gold_labels`
for downstream scoring, but the runner itself remains domain-agnostic.

Validate the external data paths before launching:

```bash
uv run python experiment-cli/experiment_cli.py validate \
  --config experiments/ministral_all_datasets.yaml
```

Run a real GPU smoke test (the limit is per dataset, and still exercises all
five variants) before submitting the full job:

```bash
uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml --rows 4
```

The command loads Ministral once, tunes each variant for each dataset, and
writes `results/ministral_all_datasets/dataset=<id>/`. The aggregate
`matrix_manifest.json` records every child manifest. Omit `--rows` for the full
datasets; ProtoEthos then reads the complete 3M-row source and should be run as
an Artemisa job with sufficient host RAM and wall time. A resumed invocation
does not duplicate completed `(variant_id, input_row_id)` rows.
