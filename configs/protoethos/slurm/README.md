# ProtoEthos Slurm job

Submit the full matrix with paths supplied at submission time:

```bash
export OUTPUT_DIR=/path/to/durable/results
export MODEL_ID=org/model-name
sbatch configs/protoethos/slurm/submit_full_matrix.sh
```

`CONFIG` defaults to `configs/protoethos/ministral_all_datasets.yaml`. Set
`DATASETS` or `VARIANTS` to run a matrix subset. The script keeps model-cache
downloads in scheduler temporary storage unless `HF_HOME` is explicitly set.
