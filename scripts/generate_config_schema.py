#!/usr/bin/env python3
"""Regenerate the machine-readable configuration schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llms_experiments.config import configuration_schema


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("docs/config.schema.json"))
    args = parser.parse_args()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **configuration_schema()}
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
