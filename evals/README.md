# LLM Inference Evaluation & Benchmarking

This directory contains evaluation cases, baseline outputs, and result thresholds for model inference pipelines.

## Measured Metrics
- **Schema Validation Rate**: Percentage of model completions matching JSON output schemas.
- **Logprob Completeness**: Presence of required token logprobs for target candidate tokens.
- **Latency & Throughput**: Requests per second and token generation latency.
