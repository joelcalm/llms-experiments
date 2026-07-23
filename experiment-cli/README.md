# Legacy v0.2 shim

`experiment_cli.py` forwards old source-checkout invocations to the installed
`llms-experiments` command and emits a deprecation warning. `run-matrix` maps to
`run`; `prepare-matrix` maps to `prepare`. The shim is scheduled for removal in
v0.3 and contains no independent inference implementation.

New integrations should use the console command and package modules documented
in the repository root README.
