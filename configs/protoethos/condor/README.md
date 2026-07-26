# ProtoEthos HTCondor jobs

`submit_extra_a100_models.sub` queues one paper-matrix job per model. Stage
this repository so `configs/protoethos/condor/run_matrix.sh` is available under
`RUN_ROOT`, then set a durable output root and submit:

```bash
export RUN_ROOT=/path/to/staged/llms-experiments
export OUTPUT_DIR=/path/to/durable/results
condor_submit "$RUN_ROOT/configs/protoethos/condor/submit_extra_a100_models.sub"
```

The wrapper selects `configs/protoethos/ministral_all_datasets.yaml` by
default, sets `model.name` and supported vLLM tuning fields from the queue row,
and defaults `HF_HOME` to the worker-local scratch directory. Set `CONFIG`,
`DATASETS`, or `VARIANTS` in the submit environment to select a different paper
configuration or matrix slice.
