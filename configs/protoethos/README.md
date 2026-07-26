# ProtoEthos run configurations

This directory is the single source of truth for configurations and scheduler
launch assets specific to the ProtoEthos paper. It contains no datasets,
results, credentials, or site-specific absolute paths.

| Asset | Purpose |
| --- | --- |
| `ministral_all_datasets.yaml` | Full multi-dataset ProtoEthos matrix |
| `mova2025.yaml` | Five-theory MoVa smoke matrix |
| `mova_full_matrix.yaml` | Full MoVa comparison matrix |
| `condor/` | HTCondor launch wrapper and paper job descriptions |
| `slurm/` | Slurm launch wrapper for the same full matrix |

Each YAML declares `config_root: ../..`, so prompt assets and relative output
paths continue to resolve from the repository root after this isolation.

Set dataset locations and durable output storage outside these files. For
example:

```bash
export DATA_ROOT=/path/to/datasets
export OUTPUT_DIR=/path/to/durable/results
llms-experiments validate configs/protoethos/ministral_all_datasets.yaml
```

Use `condor/` or `slurm/` here only for this paper. The repository-level
`condor/` and `slurm/` directories remain reusable scheduler templates.
