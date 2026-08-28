# Model Prompts Directory

This directory is reserved for repository-wide prompt assets. The runnable
experiment prompts currently live under `experiment-cli/prompt/`, and YAML
configurations refer to those Markdown and JSON assets explicitly. Keep prompt
templates and schemas separate from application code.

## Prompt Conventions

When adding a prompt asset, document:
- `identifier`: Unique string key
- `version`: Version string (e.g. `1.0.0`)
- `purpose`: Description of task
- `input_variables`: List of template variables (e.g. `{prompt}`, `{taxonomy}`)
- `expected_schema`: Expected JSON output format
