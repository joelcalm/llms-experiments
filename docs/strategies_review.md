# Implemented Processing Strategies

Strategies are YAML processor pipelines, not Python strategy subclasses.
Backends return raw text and token evidence; the stages below perform all
interpretation. This keeps the same behavior across vLLM, llama.cpp,
OpenAI-compatible endpoints, and external batch files.

## Structured single label

```yaml
processor:
  result: single_label
  stages:
    - type: json_decode
    - type: json_schema
      schema: schemas/single-label.json
      enum_from: dataset_labels
```

`json_schema` contributes a structured-output constraint to the generation
request. After generation, `json_decode` parses the text and `json_schema`
validates required fields, types, enums, bounds, and additional properties.
`enum_from` replaces string enums with the active dataset taxonomy, so the
constraint sent to the model exactly matches validation.

## Structured multi label

```yaml
processor:
  result: multi_label
  stages:
    - type: json_decode
    - type: json_schema
      schema: schemas/multi-label.json
      enum_from: dataset_labels
```

This is the same mechanism with an array-valued schema. The processor returns
the validated JSON value; thresholding and evaluation remain downstream.

## Ordinal score

```yaml
processor:
  result: ordinal_score
  stages:
    - type: json_decode
    - type: json_schema
      schema: schemas/ordinal-score.json
```

The schema limits the generated integer, typically to a Likert range. The
semantic result remains `ordinal_score`, distinct from classification.

## Categorical first-token log probabilities

```yaml
processor:
  result: categorical_logprobs
  stages:
    - type: candidate_logprobs
      candidates_from: code_labels
```

The stage compiles requirements for exactly one generated token, positional
log probabilities, and `min(20, candidate_count + 5)` alternatives. At the
first generated position it strips tokenizer whitespace and combines duplicate
spellings such as `"A"` and `" A"` with log-sum-exp. Missing candidates are
recorded as negative infinity. The parsed value and `candidate_scores` both
retain the full candidate mapping.

## Fixed yes/no probe

```yaml
processor:
  result: fixed_binary_probe
  stages:
    - type: candidate_logprobs
      candidates: ["yes", "no"]
```

This uses the same extraction algorithm with a fixed candidate set. It answers
one configured binary question and is intentionally distinct from a complete
multi-label prediction.

## Per-label soft multi-label scores

```yaml
processor:
  result: label_yes_no_logprobs
  stages:
    - type: fan_out
      over: dataset_labels
    - type: candidate_logprobs
      candidates: ["yes", "no"]
```

`fan_out` runs before generation. For each source row it creates one row per
label, sets `_target_label`, and derives a deterministic expanded
`_source_position`. Each expanded row is independently resumable and then
passes through yes/no candidate extraction. The output therefore retains the
source ID, target label, and both raw log scores.

## Verbalized confidence with token evidence

```yaml
processor:
  result: single_label_verbalized_confidence
  stages:
    - type: json_decode
    - type: json_schema
      schema: schemas/verbalized-confidence.json
      enum_from: dataset_labels
    - type: verbalized_confidence
      top_logprobs: 20
```

The model returns `label`, `confidence_tens`, and `confidence_units` as separate
schema-constrained fields. The final stage scans generated token positions
backward to align the sampled units and tens digits. At each position it
combines tokenizer spellings of digits with log-sum-exp and derives:

- literal confidence: `(10 * tens + units) / 100`;
- weighted confidence: the expected two-digit value from observed token
  probabilities, divided by 100;
- per-position digit log probabilities;
- observed digit probability mass as a coverage diagnostic.

The observed top-logprob distribution is not renormalized. Missing or
unalignable digit evidence emits a typed processing error and stops the
pipeline.

## Error behavior

Stages execute in order and stop at the first typed error. Backend failures,
JSON parsing failures, schema violations, and evidence-processing failures have
separate categories. This classification drives result status and retry:
backend failures are retryable on resume, while deterministic model validation
failures remain durable.
