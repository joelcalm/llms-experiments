# Retired Artemisa runs: provenance record

This note preserves the provenance of the two local result families removed
on 2026-08-10:

- `results/artemisa_mova_full_matrix_20260726` (113 GB before removal).
- `results/artemisa_final_matrix_20260716` (49 GB before removal).

These were stale local copies/earlier matrix runs. The current JSON-only
matrix described in `docs/protoethos_experiment_status_20260810.md` is a
separate remote run.

## Prompt contract

Both runs used the repository prompt assets, with the effective per-dataset
configuration recorded in each dataset's `effective_config.yaml` and
`manifest.json`. The exact prompt assembly was:

1. `experiment-cli/prompt/system.md`: `You are a careful text annotation
   system. Follow the requested output contract exactly.`
2. `experiment-cli/prompt/context.md`: the selected theory fragment followed
   by the selected definitions fragment.
3. A task fragment:
   - single label: `Select exactly one category from: {{labels}}. Choose the
     single category that is most present or most strongly expressed in the
     text. Do not return a second-best or co-equal category.`
   - multi label: `Select every applicable category from: {{labels}}`
4. `experiment-cli/prompt/output-json.md`: `Return JSON only. It must satisfy
   this schema:` followed by the materialised JSON schema.
5. `experiment-cli/prompt/input.md`: `Text to annotate:` followed by the
   materialised input text.

The placeholders were materialised per dataset and row. `{{labels}}` came
from the dataset configuration; theory and definitions came from the MOVA
fragments; `{{output_schema}}` came from the schema file; and `{{text}}` came
from the dataset row. The single-label schema was an object with required
string property `label`, an enum of the dataset labels, and no additional
properties. The multi-label schema was an object with required array property
`labels`, whose items were an enum of the dataset labels, and no additional
properties.

Prompt source hashes at deletion time:

| File | SHA-256 |
|---|---|
| `experiment-cli/prompt/system.md` | `bf56b5d9fe875339f6a7d6e82708d2ced53cf6db93c66d7647f111dc24f9ed23` |
| `experiment-cli/prompt/context.md` | `d97671d107389748252d1b6f6641be4ec9738086c52364c2d14261eb31438cb6` |
| `experiment-cli/prompt/task-single-label.md` | `f5087d245bcb3a7e22077bd07321bccb2f22b94b382e1d963cce2300d9bb0777` |
| `experiment-cli/prompt/task-multi-label.md` | `086762e786687044f02b0137e750fd2d7775506768773a2be2af7f4b5fa569cb` |
| `experiment-cli/prompt/output-json.md` | `afe37c29b974ca4d7458129c03bd5fc4c9b34a3e5552e41edc1aa113071b80e7` |
| `experiment-cli/prompt/input.md` | `e4fc1ea3808718d0da1bad78c3238af9d6abbad63c3fb13aefff775fd89e9eaf` |
| `experiment-cli/prompt/schema-single-label.json` | `9e00ccf2dbaae5013746e8d23451e4d1f5da92db8a8ee65fda896b549d457ba2` |
| `experiment-cli/prompt/schema-multi-label.json` | `c2b545cb47d76f93977b9afd118c63b5e7140adf191d5820a6856649ed4db718` |

MFT datasets used `mova-mft-theory.md` and `mova-mft-definitions.md`.
SHVT datasets used `mova-values20-theory.md` and
`mova-values20-definitions.md`. Cross-theory lanes used the target label
space explicitly specified in `mova_full_matrix.yaml`.

## MOVA full matrix

Source and execution records were:

- `results/artemisa_mova_full_matrix_20260726/reproducibility/mova_full_matrix.yaml`
- `results/artemisa_mova_full_matrix_20260726/reproducibility/submit_mova_full_matrix_6models.sub`
- `results/artemisa_mova_full_matrix_20260726/reproducibility/submit_mova_full_matrix_small_models.sub`
- `results/artemisa_mova_full_matrix_20260726/reproducibility/run_mova_full_matrix_worker.sh`
- `results/artemisa_mova_full_matrix_20260726/reproducibility/SHA256SUMS`

The run used `run.id: mova_full_matrix`, seed `123`, local vLLM, streaming
output, automatic batch candidates `[1, 2, 4, 8, 16, 32, 64]`, and deferred
structured-output retries with at most two attempts. It was submitted through
Condor as an 11-lane matrix: MFTC, MFRC, ValueEval, and eight ProtoEthos
lanes. Each completed dataset manifest records input rows, result rows, model
settings, prompt-group IDs, source provenance, and batch-size tuning.

The submission queues requested these model IDs:

- `google/gemma-4-31B-it`
- `Qwen/Qwen3.6-35B-A3B`
- `google/gemma-4-12b-it`
- `Qwen/Qwen3.5-9B`
- `openai/gpt-oss-20b`
- `mistralai/Ministral-3-14B-Instruct-2512`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `mistralai/Ministral-3-3B-Instruct-2512`
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`
- `meta-llama/Llama-3.2-1B-Instruct`

Observed completed manifests contained these ten model IDs, with incomplete
coverage for some: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Qwen/Qwen3.5-2B`,
`Qwen/Qwen3.5-4B`, `google/gemma-3-12b-it`, `google/gemma-4-12b-it`,
`google/gemma-4-31B-it`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`,
`meta-llama/Llama-3.2-1B-Instruct`, and
`mistralai/Ministral-3-3B-Instruct-2512-BF16`.

The recorded variants were `single_label_json`, `multi_label_json`,
`ordinal_score_json`, `single_label_code_logits`,
`independent_yes_no_logits`, `soft_multi_label_yes_no_logits`, and
`verbalized_confidence`.

Manifest row counts were:

| Dataset | Input rows | Result rows per variant |
|---|---:|---:|
| MFTC | 34,987 | 559,792 |
| MFRC | 17,886 | 286,176 |
| ValueEval | 8,865 | 230,490 |
| ProtoEthos MFT core | 200,000 | 400,000 |
| ProtoEthos SHVT core | 400,000 | 800,000 |
| ProtoEthos MFT transformed | 650,000 | 1,300,000 |
| ProtoEthos SHVT transformed | 1,300,000 | 2,600,000 |
| ProtoEthos diagnostic MFT | 450,000 | 900,000 |
| ProtoEthos diagnostic SHVT | 450,000 | 900,000 |
| ProtoEthos MFT-core cross-SHVT | 200,000 | 400,000 |
| ProtoEthos SHVT-core cross-MFT | 400,000 | 800,000 |

## FINAL matrix

The FINAL tree contained four complete model matrices, each with the same 11
dataset lanes:

- `google/gemma-4-12b-it`
- `google/gemma-4-31B-it`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.6-35B-A3B`

Its recorded variants were `single_label_json`, `multi_label_json`,
`ordinal_score_json`, `single_label_code_logits`, and
`independent_yes_no_logits`. The input row counts matched the MOVA table;
result rows were five times the input rows. Each dataset directory contained
an `effective_config.yaml` and manifest recording the local-vLLM model
settings, source-file hashes, prompt-group IDs, and batch tuning.

The separate `evaluator_metrics_20260729` directory was a derived evaluation
bundle, not raw inference output.

## Interpretation of the results

The JSON Parquet files are validated structured model outputs. A manifest is
the completion signal. The deleted raw `results.parquet` files, resume
databases, parts, logs, and errors were execution material, not needed to
interpret the requested single-label and multi-label JSON outputs once a
manifest existed.

The source prompt files, source configurations, current JSON-only status
document, and all result families outside the two retired paths remain
untouched.
