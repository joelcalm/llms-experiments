# Generic experiment CLI

`experiment_cli.py` runs local LLM experiments without embedding an application domain in Python. The YAML configuration chooses the input columns, model, prompt fragments, output schemas, candidate sets, batching policy, and result location.

## Layout

```text
experiment-cli/
  experiment_cli.py        # the only Python implementation file
  ../experiments/          # YAML experiment configurations
  data/                    # input fixtures
  prompt/                  # reusable Markdown parts and JSON schemas
```

Each support directory is one level deep. Generated `results/` are ignored by Git.

## Quick start

From the repository root, create the uv environment and run the deterministic local contract check:

```bash
UV_CACHE_DIR=.uv-cache uv sync
UV_CACHE_DIR=.uv-cache uv run python experiment-cli/experiment_cli.py gpu-preflight
UV_CACHE_DIR=.uv-cache uv run python experiment-cli/experiment_cli.py validate \
  --config experiments/local_all_modes_smoke.yaml

uv run python experiment-cli/experiment_cli.py self-test \
  --config experiments/local_all_modes_smoke.yaml
```

`self-test` uses the built-in fake backend, writes only temporary files, and checks all five request modes plus resume behaviour.

The project is locked to the CUDA 13.0 PyTorch wheels. To recreate the
environment explicitly with uv's Torch backend selector:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python \
  --torch-backend cu130 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
UV_CACHE_DIR=.uv-cache uv sync
```

`gpu-preflight` is a required GPU-installation gate: it checks both the host
driver via `nvidia-smi` and CUDA access from uv's PyTorch environment. uv
installs CUDA user-space libraries only; it cannot install the kernel-level
NVIDIA driver. A failure means the host or container runtime must expose
`/dev/nvidia*` and a compatible driver before a GPU experiment can start.

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

## CPU guardrails

`resources.cpu` makes the host-side budget part of the experiment contract.
Before vLLM is imported, the runner applies CPU affinity to itself; vLLM's
engine and worker children inherit that affinity. It also caps PyTorch, BLAS,
and tokenizer thread pools so that each child does not create another full set
of CPU workers. GPU use remains controlled by the existing
`model.gpu_memory_utilization` setting.

```yaml
resources:
  cpu:
    cores: auto          # all available CPUs except reserve_cores
    reserve_cores: 4     # keep the laptop responsive
    affinity: true
    thread_pool_size: 1  # threads per native library pool
```

Use `cores: all` only on a dedicated machine. For an explicit cap, use (for
example) `cores: 8`; it must not exceed the CPUs visible to the process. The
chosen CPU IDs, affinity status, and thread-pool environment are recorded in
every run and matrix manifest. All resource fields can be overridden without
editing YAML, for example:

```bash
uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml \
  --set resources.cpu.cores=8 --set resources.cpu.reserve_cores=0
```

The supplied Ministral matrix reserves four of the 20 logical CPUs on this
laptop and raises vLLM's GPU allocation to 95% (leaving small driver/desktop
headroom). Its prompt and schema assets are also cached in-process, so streamed
rows no longer reread Markdown files or recompute their variant hash.

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

## Batch execution with `vllm run-batch`

Prepare OpenAI Batch-compatible JSONL without loading a model:

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

`requests.jsonl` uses the vLLM/OpenAI Batch form: a unique `custom_id`, `POST`, `/v1/chat/completions`, and a request body. JSON variants carry `response_format: json_schema`; scoring variants request one token with top log-probabilities. Thus all five configured strategies can use `run-batch` in one batch job.

After the worker writes `responses.jsonl`, parse it without GPU inference:

```bash
uv run python experiment-cli/experiment_cli.py parse \
  --config experiments/local_all_modes_smoke.yaml \
  --responses results/local_all_modes_smoke/responses.jsonl
```

The parser checks response status and JSON schemas, extracts top token log-probabilities for the two scoring variants, and writes the same Parquet contract and manifest as `run`. Invalid structured responses produce `retry_requests.jsonl` when retries are enabled; submit that file with `vllm run-batch` and parse the retry response again. Final failures remain explicit rows rather than being silently dropped.

For the five-dataset matrix, prepare independent request files so separate
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
  parts/variant=<variant_id>/part-*.parquet
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
- `resources.cpu`: CPU affinity, reserved cores, and native thread-pool cap;
- `output` and `logging`: ignored local destinations.

When a source provides labels, result rows include the serialised
`gold_labels` field alongside the prediction fields.

Prompt substitutions are intentionally generic: `{{text}}`, `{{row_id}}`,
`{{dataset_id}}`, `{{labels}}`, `{{candidate_mapping}}`, `{{candidates}}`,
`{{question}}`, `{{theory}}`, `{{definitions}}`, `{{output_schema}}`,
`{{raw_response}}`, and `{{validation_errors}}`.

Prompt parts are ordinary Markdown files. A dataset can declare reusable context
in `input.prompt_parts`, for example:

```yaml
prompt_parts:
  theory: experiment-cli/prompt/theory-mft.md
  definitions: experiment-cli/prompt/definitions-mft.md
```

The `context.md` fragment ensembles those files, and each variant then adds its
task, output contract, and final `input.md` fragment. This keeps theory and
definitions in one place while allowing the same parts to be reused across JSON,
ordinal, code-logit, and yes/no variants. Static parts are rendered before the
row text so equal theory/schema prefixes can be cached by vLLM.

## Five-dataset Ministral matrix

`experiments/ministral_all_datasets.yaml` is a portable matrix for the five
datasets used in this project: MFTC, MFRC, ValueEval, and the MFT and SHVT
slices of ProtoEthos. It uses the model
`mistralai/Ministral-3-3B-Instruct-2512` and the same five interface variants
listed above. Dataset readers are format-driven: nested MFTC JSON, materialised
JSONL records, and the semicolon-delimited ProtoEthos CSV are all normalised to
the same `id`/`text` row contract. Gold labels are retained as `_gold_labels`
for downstream scoring, but the runner itself remains domain-agnostic. The
generic `alpha/beta/gamma`, `A/B/C`, and `yes/no` choices in this example are
interface-smoke defaults, not the label vocabularies of those five datasets;
define dataset-specific schemas/candidates before claiming evaluation scores.

Validate the external data paths before launching:

```bash
uv run python experiment-cli/experiment_cli.py validate \
  --config experiments/ministral_all_datasets.yaml
```

Run a real GPU smoke test (the limit is per dataset, and still exercises all
five variants) before launching the full run:

```bash
uv run python experiment-cli/experiment_cli.py run-matrix \
  --config experiments/ministral_all_datasets.yaml --rows 4
```

The command loads Ministral once, tunes each variant for each dataset, and
writes `results/ministral_all_datasets/dataset=<id>/`. The aggregate
`matrix_manifest.json` records every child manifest. Omit `--rows` for the full
datasets; ProtoEthos then streams the complete 3M-row source in bounded batches.
Each variant is appended to `parts/variant=<id>/`; a SQLite resume index prevents
duplicate `(variant_id, input_row_id, source_position)` rows and the final
Parquet files are merged without retaining the dataset in RAM.
