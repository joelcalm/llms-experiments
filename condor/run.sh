#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?Set CONFIG to a run YAML file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"

exec llms-experiments run "$CONFIG" --output "$OUTPUT_DIR"
