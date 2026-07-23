# HTCondor template

`submit.htcondor` is a deliberately minimal, site-neutral template. Install
`llms-experiments[gpu]` in the job environment, set `CONFIG` and `OUTPUT_DIR`,
and adapt resource requests to the target pool. Dataset locations and model
credentials belong in environment variables referenced by the YAML file, not
in this submit description.

Validate edits with:

```bash
bash -n condor/run.sh
```
