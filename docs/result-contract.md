# Result Contract 2.0

Every ordinary run writes `manifest.json`, `effective_config.yaml`, append-only parts, one combined result file, and one projection per variant. The extension is selected by `output.format`: `parquet`, `csv`, or `jsonl`. Matrix runs also write a matrix manifest and one independent dataset directory per lane.

A matrix may be safely split across concurrent strategy jobs with
`output.variant_isolated: true` and exactly one configured variant per job.
Such a job writes a strategy-specific manifest, effective configuration,
resume database, event log, and variant projection, while deliberately not
publishing `results.<extension>`. This prevents two jobs from clearing the
same resume index or replacing a combined file while the other strategy is
still writing. The canonical `manifest.json` is installed as a compatibility
link when absent. The final compact exporter consumes the per-variant
projections directly; a normal single-process run remains unchanged.

## Manifest

The manifest records `contract_version="2.0"`, tool version, run/model/dataset identity, the complete effective configuration, source provenance, selected batch sizes, row counts, resume counts, and resource diagnostics. Paths to logs or external batch responses are operational metadata, not row identity.

## Result Rows

Required identity fields are `contract_version`, `tool_version`, `run_id`, `model_id`, `dataset_id`, `variant_id`, `result_type`, `input_row_id`, and `source_position`. `prompt_hash`, `config_hash`, and `prompt_group_id` preserve provenance and resume compatibility.

`result_type` is semantic:

| Type | Meaning |
| --- | --- |
| `single_label` | structured single taxonomy label |
| `multi_label` | structured taxonomy label set |
| `categorical_logprobs` | complete categorical candidate log scores |
| `label_yes_no_logprobs` | one yes/no score row per target label |
| `ordinal_score` | descriptive ordinal output, not classification |
| `fixed_binary_probe` | one fixed binary question, not a full taxonomy prediction |
| `single_label_verbalized_confidence` | one taxonomy label plus literal and digit-logprob-weighted confidence |

`gold_labels` and `validation_errors` are logical string lists, and
`candidate_scores` is a logical string-to-float map. They are native Arrow
values in Parquet and losslessly encoded in CSV/JSONL. Non-finite candidate
scores use tagged encodings outside Parquet so `±Infinity` and `NaN` round-trip
without becoming strings or nulls. Variable structured output remains
canonical, key-sorted JSON in `parsed_output`; optional raw text remains in
`raw_response`. Label-wise rows retain `target_label`.

For `single_label_verbalized_confidence`, `parsed_output` contains the validated `label`, `confidence_tens`, and `confidence_units`, plus four derived fields:

- `verbalized_confidence`: the sampled two-digit percentage divided by 100.
- `logprob_weighted_confidence`: the expected percentage from the tens and units digit-token probabilities, divided by 100.
- `confidence_digit_logprobs`: the retained digit logprobs at each position.
- `confidence_digit_probability_mass`: the observed digit probability mass at each position, retained as a coverage diagnostic.

The weighted value uses model probabilities directly and does not renormalise truncated top-logprob output. Runs request the maximum 20 alternatives, while constrained decoding restricts each confidence position to a digit.

Every attempted result records status, attempts, timings, token count, batch
size, optional input text, and resource snapshot. Failed validation and
backend rows remain explicit. Backend failures are retryable on resume;
deterministic validation failures remain durable by default. A configuration
may set `validation.retry.retry_failed_validation_on_resume: true` when a
corrected prompt, schema, or token budget is being applied. Such a resume
leaves failed-validation identities retryable and the latest valid attempt
replaces the earlier row in the final projection. This is an explicit
campaign choice, not a change to the default contract semantics.

The contract contains inference results only. It defines no accuracy, F1, thresholded multi-label prediction, axiom, or aggregate evaluation field.
