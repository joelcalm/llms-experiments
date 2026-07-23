# Slurm batch path

`submit-vllm.sh` is a thin scheduler wrapper around the repository's main YAML
runner. It does not contain a second inference or evaluation implementation;
its outputs follow the same Parquet contract as local and HTCondor runs.

Install the GPU extra, set the portable run inputs, then submit:

```bash
python -m pip install 'llms-experiments[gpu]'
export CONFIG=/path/to/run.yaml
export OUTPUT_DIR=/path/to/durable/results
sbatch slurm/submit-vllm.sh
```

Set `DATASETS` or `VARIANTS` to select matrix lanes without editing the script.
Dataset paths and endpoint credentials remain configuration/environment inputs.
