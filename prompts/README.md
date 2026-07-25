# Model Prompts Directory

This directory stores versioned prompt templates, system instructions, and schema definitions separate from application code.

## Prompt Conventions

Each prompt asset should declare:
- `identifier`: Unique string key
- `version`: Version string (e.g. `1.0.0`)
- `purpose`: Description of task
- `input_variables`: List of template variables (e.g. `{prompt}`, `{taxonomy}`)
- `expected_schema`: Expected JSON output format
