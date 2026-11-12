# ProtoEthos results for the 18-model cohort

_Index to the single-label and multi-label JSON predictions used for the 18-model AIS cohort._

---

## 📌 Cohort and strategies

The cohort is the one recorded in `AIS_report/generated/current_complete_20260918/report_manifest.json`. It has 18 models and two inference strategies: `single_label_json` and `multi_label_json`. Qwen3.8-27B is included; Qwen3.5-27B and Qwen3.6-27B are not in this strict-complete cohort.

Both strategies use JSON-only responses and schema validation. The single-label prompt asks for exactly one value: the value most present or most strongly expressed in the text, with no second-best or co-equal label. The multi-label prompt asks for every applicable value, without duplicates. The source is [`configs/protoethos/protoethos_full_matrix.yaml`](../configs/protoethos/protoethos_full_matrix.yaml), with prompt text in [`task-single-label.md`](../experiment-cli/prompt/task-single-label.md), [`task-multi-label.md`](../experiment-cli/prompt/task-multi-label.md), and [`output-json.md`](../experiment-cli/prompt/output-json.md).

For ProtoEthos theory outputs, use the `protoethos_mft` and `protoethos_shvt` lanes. `mft` corresponds to Moral Foundations Theory; `shvt` corresponds to Schwartz Human Values Theory. The same strategy filenames are used in both lanes.

## 📂 File layout

For a result root containing a model directory, the four files are:

```text
<results-root>/<model-slug>/dataset=protoethos_mft/single_label_json.parquet
<results-root>/<model-slug>/dataset=protoethos_mft/multi_label_json.parquet
<results-root>/<model-slug>/dataset=protoethos_shvt/single_label_json.parquet
<results-root>/<model-slug>/dataset=protoethos_shvt/multi_label_json.parquet
```

The report manifest recorded these result roots:

```text
/lhome/ext/iiia021/iiia0214/llms_experiments_results_20260831/a100/results
/lhome/ext/iiia021/iiia0214/llms_experiments_results_20260831/h100/results
/lhome/ext/iiia021/iiia0214/llms_experiments_results_20260831/lustre_results_archive_20260918
```

These were the report’s source locations. The historical execution audit at `AIS_report/generated/current_complete_20260918/execution_status_20260918.json` records all 36 standard MFT/SHVT lanes for this 18-model cohort as complete: 7.2 million target/done prediction rows for MFT and 14.4 million for SHVT, with no incomplete lanes. The same audit reports 454 of 454 Parquet files readable across the evaluated result population. A contemporaneous storage check also records the Artemisa home backup directory `llms_experiments_results_20260831` at about 71 GiB; the similarly named folder inside the project Lustre tree was only about 60 KiB and was not the backup.

On 22 October 2026, the three report roots and the home backup directory were absent from Artemisa. A read-only search of the account’s current home tree found no Parquet files; the current project tree retains only selected later runs, partial files, and alternate-precision copies. The local `results/` tree is not a complete copy of the 18-model raw cohort. No move or deletion command was found in the available Codex operation history; the Artemisa shell history predates the backup, and no user-visible filesystem snapshot was present. Therefore the runs are confirmed complete historically, but the reason the raw backup disappeared is not established, and the complete raw cohort is not currently recoverable from the searched locations. Results retained on Artemisa are instead in experiment-specific directories below:

```text
/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716/results_json_only_*/<model-slug>/dataset=<lane>/<strategy>.parquet
```

The local repository also has older or partial copies under `results/`. Do not treat those as the 18-model cohort solely because the filenames match; check each lane’s manifest for `model_id`, revision, result status, and configuration. The cohort’s processed score export is separate from raw predictions at `AIS_report/generated/current_complete_20260918/protoethos_axiom_scores_20260920.csv`.

On Artemisa, locate matching files for a model with:

```bash
RUN_ROOT=/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716
find "$RUN_ROOT" -type f \( \
  -path '*/<model-slug>/dataset=protoethos_mft/single_label_json.parquet' -o \
  -path '*/<model-slug>/dataset=protoethos_mft/multi_label_json.parquet' -o \
  -path '*/<model-slug>/dataset=protoethos_shvt/single_label_json.parquet' -o \
  -path '*/<model-slug>/dataset=protoethos_shvt/multi_label_json.parquet' \
\) -print
```

Replace `<model-slug>` with the directory name in the table. To check a lane’s strict completion, also require its `manifest.json`, a completed checkpoint, both strategy Parquets, and no temporary output files, as defined in [`internal_documentation/INT4_RECOVERY_STATUS_20260831.md`](../internal_documentation/INT4_RECOVERY_STATUS_20260831.md).

## 🔎 Model index and live file audit

The following audit was run on 22 October 2026 over the full Artemisa shared campaign tree and the local `/home/alono/TFM` workspace. `SL+ML` means both strategy Parquets exist in that lane; `—` means no matching Parquets were found. `Exact` means the directory is the checkpoint named in the 18-model manifest; `alternate` means the local Parquets are for a different precision/checkpoint and must not be silently substituted. A lane counts as complete only when the manifest covers the full expected input population; mere Parquet presence is not completion.

The Artemisa root is `/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716`. In every listed result directory, the files are `dataset=protoethos_mft/{single_label_json,multi_label_json}.parquet` and/or `dataset=protoethos_shvt/{single_label_json,multi_label_json}.parquet`. Local paths below are relative to `/home/alono/TFM/llms-experiments`.

| Model | Manifest checkpoint / precision | Found MFT / SHVT | Exact result directory or local alternate |
|---|---|---|---|
| GPT-OSS-20B | `openai/gpt-oss-20b`; native MXFP4 | Exact: SL+ML / SL+ML | Artemisa `results_json_only_quantized_fast_a100_20260827/openai_gpt_oss_20b`; also repaired copy under `results_json_only_gpt_oss_20b_token_recovery_full_20260816_repaired_20260817/openai_gpt_oss_20b_token_recovery_full_20260816_repaired_20260817` |
| Gemma-3-12B-it | `gaunernst/gemma-3-12b-it-int4-awq`; AWQ W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/google_gemma_3_12b_it` (BF16 `google/gemma-3-12b-it`) |
| Gemma-3-27B-it | `gaunernst/gemma-3-27b-it-int4-awq`; AWQ W4A16 | — / — | No matching ProtoEthos MFT/SHVT Parquets found locally or in the searched Artemisa tree |
| Gemma-4-12B-it | `google/gemma-4-12B-it-qat-w4a16-ct`; QAT W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/google_gemma_4_12b_it` (BF16 `google/gemma-4-12b-it`) |
| Gemma-4-26B-A4B-it | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`; AWQ INT4 | Exact: SL+ML / — | Artemisa `results_json_only_async_vllm028_a100_20260828/cyankiwi_gemma_4_26b_a4b_it_qat_awq_int4` |
| Gemma-4-31B-it | `google/gemma-4-31B-it-qat-w4a16-ct`; QAT W4A16 | — / — | No matching ProtoEthos MFT/SHVT Parquets found locally or in the searched Artemisa tree |
| Gemma-4-E2B-it | `google/gemma-4-E2B-it-qat-w4a16-ct`; QAT W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/google_gemma_4_e2b_it` (BF16 `google/gemma-4-E2B-it`) |
| Gemma-4-E4B-it | `google/gemma-4-E4B-it-qat-w4a16-ct`; QAT W4A16 | Exact: SL+ML / SL+ML | Artemisa `results_json_only_async_vllm028_a100_20260828/google_gemma_4_e4b_it_qat_w4a16_ct` |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B-Instruct`; BF16 | Exact: SL+ML / SL+ML | Artemisa `results_json_only_async_vllm028_a100_20260828/meta_llama_3_2_1b_instruct`; local copy `results/artemisa_sync_20260810b/meta_llama_3_2_1b_instruct` |
| Ministral-3-14B | `cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit`; AWQ W4A16 | Exact: SL+ML / SL+ML | Artemisa `results_json_only_quantized_fast_a100_20260827/cyankiwi_ministral_3_14b_instruct_2512_awq_4bit` |
| Ministral-3-3B | `cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit`; AWQ W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/mistralai_ministral_3_3b_bf16` (BF16 checkpoint); another local full run is `results/ministral_full_local_20260723_gpu` |
| Muse-Glimmer-30B | `RedHatAI/Muse-Glimmer-30B-W4A16`; compressed-tensors W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_recovered_20260820/results_json_only_fp8_muse_native_container_full_20260814_repaired_20260817/redhatai_muse_glimmer_30b_fp8_block_native_container_full_20260814_repaired_20260817` (FP8, not W4A16) |
| Qwen3.5-2B | `QuantTrio/Qwen3.5-2B-AWQ`; AWQ W4A16 | Exact: SL+ML / SL+ML | Artemisa `results_json_only_async_vllm028_a100_20260828/quanttrio_qwen3_5_2b_awq` |
| Qwen3.5-35B-A3B | `QuantTrio/Qwen3.5-35B-A3B-AWQ`; AWQ W4A16 | Smoke only (64 input rows per strategy) / same | Artemisa `results_json_only_protoethos_int4_validation_20260826/qwen_qwen3_5_35b_a3b_awq` |
| Qwen3.5-4B | `QuantTrio/Qwen3.5-4B-AWQ`; AWQ W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/qwen_qwen3_5_4b` (BF16 `Qwen/Qwen3.5-4B`) |
| Qwen3.5-9B | `QuantTrio/Qwen3.5-9B-AWQ`; AWQ W4A16 | Exact: — / —; alternate: SL+ML / SL+ML | Local `results/artemisa_sync_20260810b/qwen_qwen3_5_9b` (BF16 `Qwen/Qwen3.5-9B`) |
| Qwen3.6-35B-A3B | `QuantTrio/Qwen3.6-35B-A3B-AWQ`; AWQ W4A16 | Smoke only (64 input rows per strategy) / same | Artemisa `results_json_only_protoethos_int4_validation_20260826/qwen_qwen3_6_35b_a3b_awq_quanttrio` |
| Qwen3.8-27B | `philbert440/Qwen3.8-27B-W4A16-AWQ`; AWQ W4A16 | Exact: — / —; alternate FP8: SL+ML / — | Local `results/artemisa_recovered_20260820/results_json_only_fp8_qwen38_full_20260818/qwen_qwen3_8_27b_fp8_full_20260818` |

Thus, exact-checkpoint full outputs currently exist for five models across both theories (20 of 72 theory-by-strategy outputs), and Gemma-4-26B-A4B has its full MFT lane (two more outputs). The Qwen3.5/3.6-35B-A3B files are only 64-row smoke runs and do not count as complete. Several other models have full local BF16 alternatives, and Muse/Qwen3.8 have FP8 alternatives; these are not the checkpoints named in the manifest and must be treated separately. Overall, 50 of the 72 exact-checkpoint theory-by-strategy outputs are missing as full lanes. The local `AIS_report/generated/current_complete_20260918/protoethos_axiom_scores_20260920.csv` is a derived metrics export, not the missing raw Parquets.

## 🧾 Cohort metadata and provenance

The authoritative 18-model list, checkpoint IDs, precision labels, result roots, strategy names, and completion rule are in `AIS_report/generated/current_complete_20260918/report_manifest.json`. The count of 18 refers to the report’s strict-complete cohort snapshot. The current filesystem search above establishes where corresponding raw strategy files can be found now; it does not certify result content or current lane completion.
