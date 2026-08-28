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

## Model-specific JSON compatibility

The quantised worker passes optional model controls through to the common
configuration. For checkpoints whose chat/output protocol is incompatible with
vLLM guided JSON, set `STRUCTURED_OUTPUTS=false` and, when required by the
checkpoint, `GENERATION_CONFIG=auto`. This disables only generation-time
constraints; the normal JSON parser and schema validator remain active.
