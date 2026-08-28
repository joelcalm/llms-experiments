# ProtoEthos json_only experiment matrix — status as of 2026-08-10T17:50+02:00

> ⚠️ **Historical snapshot:** This document records the 10 August state and
> is retained for provenance. It is not a current job or completion report.
> Check [`internal_documentation/00-live-status.md`](../internal_documentation/00-live-status.md)
> and query Artemisa before using any status below.

Run root (Artemisa Lustre): `/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716`

## Where to find results

- General benchmarks (MFTC, MFRC, ValueEval): `results_json_only_general/<model_slug>/dataset=<mftc|mfrc|valueeval>/`
- ProtoEthos (8 lanes, full 3M-row deliverable): `results_json_only_protoethos/<model_slug>/dataset=<lane>/`
- Each dataset directory contains `single_label_json.parquet`, `multi_label_json.parquet`, `manifest.json` (present only once BOTH variants for that dataset are complete), plus `results.parquet`, `.resume.sqlite`, `errors/`, `logs/`.
- **A dataset counts as done only when `manifest.json` exists** — that's the authoritative completion signal, not just a `.parquet` file being present (which can exist mid-run under `parts/`).
- The 8 ProtoEthos lanes: `protoethos_mft`, `protoethos_shvt`, `protoethos_mft_transformed`, `protoethos_shvt_transformed`, `protoethos_diagnostic_mft`, `protoethos_diagnostic_shvt`, `protoethos_mft_core_cross_shvt`, `protoethos_shvt_core_cross_mft`.
- Full chronological log of every fix, root-cause, and check made during this monitoring run: `internal_documentation/03-protoethos-monitoring-log.md` on the cluster.

## Fully complete — all lanes (9 of 19 models)

General (3/3) + full ProtoEthos (8/8), both `single_label_json` and `multi_label_json`:

| Model slug | Model ID |
|---|---|
| `google_gemma_3_12b_it` | google/gemma-3-12b-it |
| `google_gemma_4_12b_it` | google/gemma-4-12b-it |
| `google_gemma_4_e2b_it` | google/gemma-4-E2B-it |
| `google_gemma_4_e4b_it` | google/gemma-4-E4B-it |
| `meta_llama_3_2_1b_instruct` | meta-llama/Llama-3.2-1B-Instruct |
| `mistralai_ministral_3_3b_bf16` | mistralai/Ministral-3-3B-Instruct-2512-BF16 |
| `qwen_qwen3_5_2b` | Qwen/Qwen3.5-2B |
| `qwen_qwen3_5_4b` | Qwen/Qwen3.5-4B |
| `qwen_qwen3_5_9b` | Qwen/Qwen3.5-9B |

## Partially complete

| Model slug | General | ProtoEthos | Notes |
|---|---|---|---|
| `google_gemma_4_31b_it` | 0/3 | 8/8 | ProtoEthos done, general lanes not started |
| `qwen_qwen3_5_35b` | 0/3 | 8/8 | ProtoEthos done, general lanes not started |
| `qwen_qwen3_6_35b_a3b` | 3/3 | 0/8 | General done; ProtoEthos job queued, waiting on an H100 slot |
| `qwen_qwen3_5_27b` | 0/3 | 3/8 | In progress; job resubmitted after hitting the site's 48h wall-clock hold |

## Not started

`google_gemma_3_27b_it`, `google_gemma_4_26b_a4b` (chat-template fix deployed and queued, not yet validated), `meta_llama_4_scout_17b_16e_instruct`, `mistralai_ministral_3_14b`, `openai_gpt_oss_20b`.

All are queued as production jobs on H100; none held or crash-looping as of this writing. Bottleneck is cluster-wide H100 contention (Artemisa fair-share negotiation against other users), not a code/config defect.

## Out of scope / superseded

- `cuongpp_gemma_3_12b_it_gptq_4bit` — dropped from the matrix (user decision); its non-quantized base `google_gemma_3_12b_it` (already complete, see above) serves as the replacement.
- `mistralai_ministral_3_3b` (fp8 checkpoint) — superseded by `mistralai_ministral_3_3b_bf16`. The fp8 slug will stay at 0/0 permanently; that's expected, not a bug.

## Fixes applied during this monitoring run (full detail in the monitoring log)

1. **Ministral-3-3B fp8 checkpoint crash** — root-caused to `quant_method: fp8` baked into the checkpoint weights, incompatible with H100's Cutlass kernel in this vLLM build. Fixed by swapping to the vendor's non-quantized `-BF16` sibling repo, targeted at A100. Fully validated end-to-end.
2. **gemma-4-26B-A4B chat-template crash** — vLLM's `.chat()` resolution never reads back the `chat_template` passed to `LLM()`'s constructor; fixed by loading the template once and passing it explicitly on every `.chat()` call. Deployed to both local source and remote checkout; validation still pending an H100 slot.
3. **Site-wide 48h job wall-clock cap** — recurring hold pattern; `condor_qedit` does not work around it (Condor recomputes from job-start bookkeeping). Fix pattern: `condor_rm` + fresh resubmit of just the affected model, relying on checkpoint/resume.
4. **A100 viability for quantized models** — confirmed via two independent tests (`openai/gpt-oss-20b`, `mistralai/Ministral-3-14B-Instruct-2512`) that this vLLM build is missing the `_C::gptq_marlin_repack` compiled CUDA kernel needed to run any fp8/GPTQ/mxfp4-quantized model on non-H100 hardware. A100 only works for genuinely bf16 checkpoints. Not fixable without a vLLM rebuild.
5. **Self-inflicted, cosmetic-only bug** — a worker-script edit landed on disk while a job was still executing it, corrupting only its trailing "write completed status" step (not the actual data). One model (`qwen_qwen3_5_35b`) exited non-zero despite having fully correct output; confirmed via manifest presence.
