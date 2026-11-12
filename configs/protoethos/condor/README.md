# ProtoEthos HTCondor jobs

This directory holds HTCondor job definitions for ProtoEthos campaigns. The
**per-experiment `*.sub` and `run_*_worker.sh` scripts are runtime artifacts and
are git-ignored**; only the documentation and shared templates below are tracked.
Regenerate them from a configuration with the workflow described here, or keep
them in a separate, non-committed working tree.

## Template example

The repository ships a minimal, tracked submit template at `condor/submit.htcondor`
and a worker wrapper at `condor/run.sh`. Adapt it for a paper matrix:

```text
# condor/submit.htcondor (tracked template)
universe = vanilla
executable = condor/run.sh
request_cpus = 16
request_memory = 96GB
request_gpus = 1
environment = "CONFIG=$(CONFIG) OUTPUT_DIR=$(OUTPUT_DIR)"
output = logs/llms-experiments.$(ClusterId).$(ProcId).out
error  = logs/llms-experiments.$(ClusterId).$(ProcId).err
log    = logs/llms-experiments.$(ClusterId).log
queue 1
```

```bash
# condor/run.sh runs the tracked CLI; pass a durable OUTPUT_DIR.
export CONFIG=configs/protoethos/protoethos_full_matrix.yaml
export OUTPUT_DIR=<durable-results>
condor_submit condor/submit.htcondor
```

`condor/run.sh` executes `llms-experiments run "$CONFIG" --output "$OUTPUT_DIR"`,
so the worker has no second inference implementation and produces the same Parquet
contract as local and Slurm runs.

For a matrix split across many queue rows, write a per-model `*.sub` that sets
`CONFIG`, `DATASETS`, and `VARIANTS` in the submit `environment` and points
`executable` at a worker script that calls the CLI with `output.variant_isolated:
true`. Keep those generated descriptors out of the repository.

## H100 preflight checklist

The ProtoEthos H100 worker must receive an expanded absolute `DATA_ROOT`, not
the literal string `${DATA_ROOT}`. Before submitting a generated descriptor,
inspect its rendered environment with `condor_submit -dry-run` or the first
`condor_history -long` record and verify that it contains, at minimum:

```text
DATA_ROOT=/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716/mova_data
CONFIG_PATH=/lustre/ific.uv.es/ml/iiia021/llms_experiments_full_matrix_20260716/code_quantized_20260826/configs/protoethos/protoethos_full_matrix.yaml
```

The worker performs the same checks itself and runs the CLI configuration
validator before downloading the model. A missing, unresolved, relative, or
non-existent data root is recorded as `configuration_failed` and releases the
job without starting vLLM. The current H100 descriptor uses a 24-hour
`PeriodicRemove` boundary; streamed Parquet parts, checkpoints, manifests, and
resume databases must therefore remain on durable storage so a recycle can
resume without duplicating work. Do not extend that boundary without an
explicit campaign decision.

## Model-specific JSON compatibility

The quantised worker passes optional model controls through to the common
configuration. For checkpoints whose chat/output protocol is incompatible with
vLLM guided JSON, set `STRUCTURED_OUTPUTS=false` and, when required by the
checkpoint, `GENERATION_CONFIG=auto`. This disables only generation-time
constraints; the normal JSON parser and schema validator remain active.
