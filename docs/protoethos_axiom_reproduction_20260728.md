# Reproducing the 28 July 2026 Artemisa ProtoEthos evaluation

The evaluation uses the published `protoethos-evaluator` library to score all
nine Chapter 3 axioms. It does not reuse the older provisional axiom JSON.

## Required inputs

For each of the four completed models (`google_gemma_4_31b_it`,
`google_gemma_4_12b_it`, `qwen_qwen3_5_9b`, and
`qwen_qwen3_6_35b_a3b`), retain the completed `manifest.json` and
`effective_config.yaml` plus these three prediction Parquets for each lane:

```text
dataset=protoethos_mft/
dataset=protoethos_mft_transformed/
dataset=protoethos_diagnostic_mft/
dataset=protoethos_shvt/
dataset=protoethos_shvt_transformed/
dataset=protoethos_diagnostic_shvt/

{multi_label_json,single_label_json,single_label_code_logits}.parquet
```

That is 72 raw prediction Parquets, 24 manifests, and 24 effective
configurations. Exclude `independent_yes_no_logits`, `ordinal_score_json`,
partial `part-*.parquet`, cross-theory lanes, and the unsynchronised soft
multilabel snapshot: they are not full-taxonomy inputs to this evaluation.

The canonical controlled-case source is
`protoethos/EIT/deliverable/dataset/dataset_3m.csv`. For a compact re-run
bundle, include the six derived metadata Parquets created below instead:

```text
metadata/{mft,shvt}_{core,transformed,deontic}.parquet
```

## Run

```bash
cd protoethos-evaluator
UV_CACHE_DIR=.uv-cache uv sync --group dev --extra io
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=.uv-cache uv run pytest -q
UV_CACHE_DIR=.uv-cache uv run python scripts/run_artemisa_axioms.py \
  --dataset ../protoethos/EIT/deliverable/dataset/dataset_3m.csv \
  --results-root ../llms-experiments/results/artemisa_final_matrix_20260716 \
  --output ../llms-experiments/results/artemisa_final_matrix_20260716/evaluator_metrics_20260728
```

The run writes `axiom_scores.csv`, `run_manifest.json`, the six metadata
Parquets, and complete per-profile JSON audits. `run_manifest.json` freezes the
exact Chapter 3 affinities: MFT has only the five virtue--vice `-1` pairs;
SHVT has only the 20 adjacent `0.95` links and ten diametric `-1` pairs. All
other off-diagonal entries are zero.

## Upload set

The compact, reproducible results release must include:

1. `axiom_scores.csv` and `run_manifest.json`;
2. `metadata/{mft,shvt}_{core,transformed,deontic}.parquet`;
3. the 72 raw prediction Parquets; and
4. the matching manifests and effective configurations.

Include `dataset_3m.csv` (or its versioned dataset package and checksum) for
independent reconstruction of the metadata. The 120 detailed report JSON files
are an optional audit archive: they are large but can always be regenerated
from the compact release inputs.
