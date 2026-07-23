# Changelog

## 0.2.0 — 2026-07-23

First installable release. Adds the modular package and five-command console
CLI, typed configuration schema, fake/endpoint/in-process/external backends,
label-wise soft multi-label inference, result contract 2.0, atomic resume
publication, CPU-light default dependencies, vLLM 0.25.1 GPU extra, portable
scheduler templates, and release-quality documentation and CI. The source-path
shim remains for v0.2 and is scheduled for removal in v0.3.

On SM 12.x devices, v0.2 also disables only the incompatible FlashInfer sampler
by default; explicit vLLM environment configuration remains authoritative.
