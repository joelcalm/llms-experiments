# Result contract 2.0

Every run writes `manifest.json`, `effective_config.yaml`, append-only parts,
`results.parquet`, and one projection per variant. Matrix runs also write a
matrix manifest and one independent dataset directory per lane.

## Manifest

The manifest records `contract_version="2.0"`, tool version, run/model/dataset
identity, the complete effective configuration, source provenance, selected
batch sizes, row counts, resume counts, and resource diagnostics. Paths to logs
or external batch responses are operational metadata, not row identity.

## Result rows

Required identity fields are `contract_version`, `tool_version`, `run_id`,
`model_id`, `dataset_id`, `variant_id`, `result_type`, `input_row_id`, and
`source_position`. `prompt_hash`, `config_hash`, and `prompt_group_id` preserve
provenance and resume compatibility.

`result_type` is semantic:

| Type | Meaning |
| --- | --- |
| `single_label` | structured single taxonomy label |
| `multi_label` | structured taxonomy label set |
| `categorical_logprobs` | complete categorical candidate log scores |
| `label_yes_no_logprobs` | one yes/no score row per target label |
| `ordinal_score` | descriptive ordinal output, not classification |
| `fixed_binary_probe` | one fixed binary question, not a full taxonomy prediction |

`gold_labels` and `validation_errors` are native Arrow lists.
`candidate_scores` is a native Arrow map of string to float. Variable structured
output remains canonical, key-sorted JSON in `parsed_output`; optional raw text
remains in `raw_response`. Label-wise rows retain `target_label`.

Every attempted result records status, attempts, timings, token count, batch
size, optional input text, and resource snapshot. Failed validation and backend
rows remain explicit. Backend failures are retryable on resume; deterministic
validation failures remain durable.

The contract contains inference results only. It defines no accuracy, F1,
thresholded multi-label prediction, axiom, or aggregate evaluation field.
