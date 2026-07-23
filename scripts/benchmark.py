#!/usr/bin/env python3
"""Developer-only throughput benchmark; not part of the supported CLI."""

from __future__ import annotations

import argparse
import json

from llms_experiments._core import benchmark
from llms_experiments.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--approaches", help="Comma-separated api,run-batch,python subset")
    parser.add_argument("--rows", type=int)
    args = parser.parse_args()
    approaches = [item.strip() for item in args.approaches.split(",") if item.strip()] if args.approaches else None
    print(json.dumps(benchmark(load_config(args.config), approaches, args.rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
