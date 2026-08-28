# LLM Inference Evaluation & Benchmarking

This directory contains evaluation thresholds and, when added, small
version-controlled evaluation fixtures. Large model outputs and downloaded
datasets belong outside Git and are referenced through manifests.

## Measured Metrics
- **Schema validation rate**: Percentage of completions matching the configured JSON schema.
- **Logprob completeness**: Presence of required token log probabilities for target candidates.
- **Latency and throughput**: Requests per second and token-generation latency.

The checked-in [`thresholds.toml`](thresholds.toml) stores inference acceptance
thresholds. Predictive accuracy and ProtoEthos axiom metrics are computed by
the separate `protoethos-evaluator` project, not by this directory.
