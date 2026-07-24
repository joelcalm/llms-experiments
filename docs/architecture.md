# Architecture

The package is organized around explicit interfaces and factory maps:

```text
YAML configuration
  -> InputReader -> normalized rows
  -> Processor.prepare_rows() -> request rows
  -> Processor.requirements -> GenerationRequest
  -> Backend.generate() -> RawModelResponse
  -> Processor.process() -> ProcessedResult
  -> OutputStore -> durable parts and final projections
```

The important boundary is between `RawModelResponse` and `ProcessedResult`.
Backends only transport prompts and normalize model evidence: generated text,
token positions, alternative-token log probabilities, token count, metadata,
and transport failures. Processors own all semantic behavior: fan-out, parsing,
schema validation, candidate aggregation, confidence extraction, enrichment,
and the semantic result type.

## Package layout

| Package/module | Responsibility |
| --- | --- |
| `backends/` | `Backend` ABC and vLLM, llama.cpp, OpenAI-compatible, and fake transports |
| `inputs/` | `InputReader` ABC and normalized, stable-position source readers |
| `processors/` | typed contracts, `ProcessingStage` ABC, pipeline compiler, and built-in stages |
| `outputs/` | `OutputStore`/`ResultFileWriter` ABCs, Parquet, CSV, JSONL, atomic parts, and resume index |
| `configuration.py` | YAML loading, validation, overrides, and dataset lanes |
| `prompting.py` | prompt assets, rendering, schemas, hashes, and external request bodies |
| `orchestration.py` | bounded run loop, tuning, backoff, retry, resume, and matrices |
| `batching.py` | external batch preparation and response ingestion |
| `results.py` | format-independent result contract and row construction |
| `events.py` | structured events and diagnostic error streams |
| `_core.py` | deprecated compatibility re-exports only |

Imports point inward toward contracts. In particular, processors do not import
backends or output formats, and backends do not know which semantic strategy
will consume their response.

## Processor pipeline

A processor is compiled once for each variant. During compilation, ordered
stages declare their `ResponseRequirements`; the processor merges them and
rejects conflicts such as structured JSON plus one-token candidate scoring.
The resulting `GenerationRequest` is identical for in-process, HTTP, llama.cpp,
and external-batch execution.

Row-preparation stages must come first. Response stages then receive immutable
`ProcessingState`. A typed `ProcessingError` stops the pipeline immediately,
which prevents later enrichment from running against invalid state. The final
`ProcessedResult` contains the semantic result type, value, optional candidate
scores, target label, errors, and processor metadata.

The built-in stages are:

- `identity`: preserve raw model text;
- `fan_out`: clone each source row over `dataset_labels` with stable identities;
- `json_decode`: parse generated JSON;
- `json_schema`: constrain and validate the parsed value;
- `candidate_logprobs`: aggregate first-position token spellings with log-sum-exp;
- `verbalized_confidence`: align two generated digits and derive literal and weighted confidence.

## Adding an extension

Extensions are intentionally explicit: implement the abstract class, then add
one entry to its factory map.

```python
from llms_experiments.backends import Backend
from llms_experiments.processors import GenerationRequest, RawModelResponse

class MyBackend(Backend):
    def generate(self, prompts, request: GenerationRequest):
        return [RawModelResponse(text="...", token_count=1) for _ in prompts]

    def close(self):
        ...
```

Register it in `BACKEND_TYPES` in `backends/factory.py`. Input readers follow
the same pattern with `InputReader.iter_rows()` and `INPUT_READER_TYPES`.
Output formats implement `OutputStore.open_writer()` and `iter_file()`, then
join `OUTPUT_STORE_TYPES`. Processing stages implement `requirements()` and
`process()` (or `prepare_rows()`), then join `STAGE_TYPES`.

Factory registration is local and reviewable. There is no dynamic import or
entry-point discovery.

## Durability and resume

Every output store uses the same logical schema. Rows are buffered into
append-only parts and published by atomic rename before their identities are
committed to SQLite. Resume seeds only from durable rows whose configuration
hash matches. A backend failure remains retryable; deterministic validation
failures remain complete. Final per-run and per-variant files are projections
of the immutable parts, with the newest retry winning.

Parquet is the default. CSV encodes each contract cell as JSON, and JSONL uses
tagged values for non-finite floats, so nulls, lists, maps, and `±Infinity`
round-trip without changing the logical row.

External batch preparation emits OpenAI Batch-shaped JSONL. Its parser builds
the same `RawModelResponse`, executes the same processor, and writes through
the same output store as the in-process runner.
