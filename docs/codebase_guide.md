# Codebase Architecture and Guide

This guide explains the design, components, configurations, and scripts in the `llms-experiments` repository. It is intended to help developers and researchers understand the codebase, customize experiments, and run them efficiently.

---

## 1. Project Directory Structure

Here is a bird's-eye view of the repository layout:

```text
llms-experiments/
  experiment-cli/
    experiment_cli.py        # Main execution CLI (runs validation, inference, and benchmarks)
    README.md                # Fast setup and guide for the experiment CLI
    data/
      smoke.parquet          # Smoke test input dataset
    prompt/                  # Reusable prompt markdown fragments and JSON schemas
      system.md
      context.md
      task-*.md
      schema-*.json
  experiments/               # YAML experiment configurations
    local_all_modes_smoke.yaml
    ministral_all_datasets.yaml
  docs/                      # Auxiliary documentation
    codebase_guide.md        # This file
    strategies_review.md     # In-depth review of classification strategies
  condor/                    # HTCondor batch path (site-agnostic, paths from RUN_ROOT)
    submit_*.sub             # Condor submit files, one queue per model matrix
    run_matrix_worker.sh     # Per-slot entry point: runs experiment_cli.py run-matrix
    submit_when_ready.sh     # Orchestrator: submits once the environment is staged
    README.md                # RUN_ROOT contract and expected layout
  slurm/                     # Slurm cluster batch-inference path
    submit-vllm.sh           # Slurm cluster scheduler submission script
    batch_infer_simple.py    # Batch inference script (driven by submit-vllm.sh)
    sample_csv_deterministic.py # Dataset sampling script
    prompts/                 # One-label prompt templates (mft/shvt_one_label.txt)
  scripts/legacy/            # Older standalone single-sentence prototypes
    annotate.py              # Standalone sentence annotation script
    run_vllm.py              # Standalone vLLM server integration script
  prompts_legacy/            # Archived demo prompt examples
  tests/                     # pytest suite (fake backend, no GPU)
    golden/                  # snapshots pinning result rows + manifests
  pyproject.toml             # Python dependencies & tool settings (uv.lock is the lockfile)
```

---

## 2. Core Execution CLI (`experiment_cli.py`)

The heart of the repository is [experiment-cli/experiment_cli.py](file:///home/alono/TFM/llms-experiments/experiment-cli/experiment_cli.py). It operates entirely dynamically based on a YAML configuration and Markdown/JSON assets, isolating application-specific logic from Python.

### CLI Subcommands

You can run `experiment_cli.py` with several subcommands:

1. **`validate`**
   * **Purpose**: Performs a dry-run check of the configuration file. It verifies that top-level keys (`run`, `model`, `variants`, `output`) are present, paths to prompt assets exist, schemas compile, and that batch size boundaries are valid.
   * **Usage**:
     ```bash
     uv run python experiment-cli/experiment_cli.py validate --config experiments/local_all_modes_smoke.yaml
     ```

2. **`run`**
   * **Purpose**: Runs the full experiment pipeline sequentially across all configured variants.
   * **Backend**: Uses either `local_vllm` (direct in-process GPU execution) or `fake` (for testing/dry-runs).
   * **Resuming**: It keeps a manifest and resume index. If aborted mid-run, re-running the same configuration skips already processed rows (by mapping `(variant_id, input_row_id)`).

3. **`run-matrix`**
   * **Purpose**: Runs a single dataset lane from a multi-dataset config, enabling parallel launches across multiple workers.
   * **Usage**:
     ```bash
     uv run python experiment-cli/experiment_cli.py run-matrix --config experiments/ministral_all_datasets.yaml --datasets mftc
     ```

4. **`prepare` / `prepare-matrix`**
   * **Purpose**: Prepares batch request files in OpenAI batch-compatible JSONL format (`requests.jsonl`) without loading a model.
   * **Usage**:
     ```bash
     uv run python experiment-cli/experiment_cli.py prepare --config experiments/local_all_modes_smoke.yaml
     ```

5. **`batch-command`**
   * **Purpose**: Prints the exact terminal command required to run the prepared `requests.jsonl` file using the external `vllm run-batch` CLI.

6. **`self-test`**
   * **Purpose**: Uses the `fake` backend to execute all five request modes on temporary output directories to verify that parsing, validation, and resuming behave correctly.

7. **`benchmark`**
   * **Purpose**: Compares inference performance across three execution paths:
     1. `python` (direct in-process `LLM.chat`)
     2. `api` (OpenAI-compatible server endpoint)
     3. `run-batch` (external batch CLI)
   * It logs latency, GPU snapshots, tokens/second, and requests/second, saving results in `benchmark.json`.

---

## 3. Configuration Contract (YAML)

All configuration variables are managed via YAML files (e.g., [experiments/local_all_modes_smoke.yaml](file:///home/alono/TFM/llms-experiments/experiments/local_all_modes_smoke.yaml)). Key sections include:

* **`run`**: Contains metadata, random seeds, class labels, code mappings, and the binary question.
* **`input`**: Declares path, format (Parquet/CSV), text columns, and unique IDs.
* **`model`**: Identifies the model identifier, backend (e.g. `local_vllm`), memory parameters, and optional GPU limits.
* **`resources`**: Controls core allocation (`cores: auto`) and caps PyTorch/BLAS thread pools to `1` per worker.
* **`variants`**: Configures the individual experiments (e.g. JSON validation rules, candidate sets, request modes).
  A variant's `prompts` are concatenated into the user turn. An optional
  `system_prompt` (one path or a list) is rendered into a separate system turn
  and sent by every backend. Because it is instructions rather than data, it is
  rendered once per variant, so it may use `{{labels}}` or `{{output_schema}}`
  but not `{{text}}`/`{{row_id}}` — those raise rather than render empty.
* **`batch`**: Defines batch sizing policy (such as `mode: auto`, starting batch candidates, and halving on failure).

---

## 4. The 5 Inference & Classification Strategies

The codebase defines 5 variants that evaluate classifications differently:

| Variant | Mode | Outputs | Constraint Handling |
| :--- | :--- | :--- | :--- |
| **`single_label_json`** | `generate` | Multi-class label | vLLM Guided JSON Schema (`schema-single-label.json`) |
| **`multi_label_json`** | `generate` | Multiple labels | vLLM Guided JSON Schema (`schema-multi-label.json`) |
| **`ordinal_score_json`**| `generate` | Numeric rating | vLLM Guided JSON Schema (`schema-ordinal-score.json`) |
| **`single_label_code_logits`** | `candidate_logprobs` | Multi-class scores | Raw first-token logprobs aggregated by whitespace-stripping |
| **`independent_yes_no_logits`** | `candidate_logprobs` | Binary scores | Raw first-token logprobs aggregated for `["yes", "no"]` |

### Logprob Aggregation Improvement
When running in `candidate_logprobs` mode:
* The runner restricts model generation to `max_completion_tokens: 1` and fetches top token log-probabilities.
* Tokenizers often split candidates into multiple token representations (e.g. `" A"` vs. `"A"`).
* **Our optimization**: The codebase uses `aggregate_candidate_logprobs` to group raw tokens by stripping leading/trailing whitespace, and performs a mathematically correct probability summation using `logsumexp`:
  $$\text{logprob}_{\text{aggregated}} = \text{max\_lp} + \log\left(\sum e^{\text{logprob} - \text{max\_lp}}\right)$$
* It also caps `top_logprobs` at `20` to guarantee strict OpenAI API compatibility and avoid validation errors on external endpoints.

---

## 5. Performance Optimizations

This framework features production-ready optimizations for high-throughput GPU workloads:
1. **Automatic Prefix Caching**: Enabled on vLLM to skip KV-cache computation on shared prompt prefixes.
2. **CPU Affinity and Thread Containment**: Caps PyTorch, BLAS, and tokenizer threads (`thread_pool_size: 1`) to prevent thrashing.
3. **Dynamic Batch Sizing**: Finds the highest stable throughput and automatically halves the batch size if an OOM occurs.
4. **Markdown/Asset Caching**: Static fragments are cached in memory to minimize file reads during row streaming.

---

## 6. Secondary / Standalone Scripts

Apart from the primary CLI, the codebase includes several utility scripts. The
two cluster schedulers live in separate folders — HTCondor in `condor/`, Slurm
in `slurm/` — so each cluster's jobs are submitted independently.

**HTCondor cluster path (`condor/`):**

* **`condor/submit_*.sub`**: HTCondor submit files. Each queues one
  `experiment_cli.py run-matrix` worker per model (`condor_submit condor/<file>.sub`). Paths resolve
  from `RUN_ROOT` at submit time, so the files carry no site-specific details.
* **`condor/run_matrix_worker.sh`**: The per-slot entry point a `.sub` executes; it drives one full
  matrix through `experiment_cli.py` on a single GPU slot.
* **`condor/submit_when_ready.sh`**: Waits for the one-time environment setup to finish, then
  submits the matrix.
* **`condor/README.md`**: The `RUN_ROOT` contract and the expected directory layout.

**Slurm cluster batch path (`slurm/`):**

* **`slurm/submit-vllm.sh`**: A cluster scheduling helper script to request GPU allocations, set up virtual environments, and trigger batch experiments. Run it with `sbatch slurm/submit-vllm.sh`; its companion scripts and prompt templates ship alongside it under `slurm/`.
* **`slurm/batch_infer_simple.py`**: Streamlined GPU classifier that reads a CSV row-by-row, batches prompts, runs a single-label classification model, and outputs a Parquet table.
* **`slurm/sample_csv_deterministic.py`**: Draws a reproducible deterministic sample of rows from a CSV by text column.

**Legacy single-sentence prototypes (`scripts/legacy/`):**

* **`scripts/legacy/annotate.py`**: A specialized WSL-compatible script that uses vLLM to read a CSV dataset of sentences, apply guided JSON schemas, and produce moral/value labels (`results_mft_0.8b.jsonl`).
* **`scripts/legacy/run_vllm.py`**: Queries a local vLLM API server and formats prompts to retrieve moral/value scores via standard OpenAI library calls.
