# Slurm batch path

`submit-vllm.sh` is a thin scheduler wrapper around the repository's main YAML
runner. It does not contain a second inference or evaluation implementation;
its outputs follow the same Parquet contract as local and HTCondor runs.

## Template example

The tracked `slurm/submit-vllm.sh` is the reference template:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=llm-inference
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${CONFIG:?Set CONFIG to a run YAML file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"

args=(llms-experiments run "$CONFIG" --output "$OUTPUT_DIR")
[[ -n "${DATASETS:-}" ]] && args+=(--datasets "$DATASETS")
[[ -n "${VARIANTS:-}" ]] && args+=(--variants "$VARIANTS")

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
srun "${args[@]}"
```

Install the GPU extra, set the portable run inputs, then submit:

```bash
uv sync --all-groups --extra gpu
export CONFIG=<run.yaml>
export OUTPUT_DIR=<durable-results>
sbatch slurm/submit-vllm.sh
```

Set `DATASETS` or `VARIANTS` to select matrix lanes without editing the script.
Dataset paths and endpoint credentials remain configuration/environment inputs.

