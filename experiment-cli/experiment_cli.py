#!/usr/bin/env python3
"""Deprecated v0.2 forwarding shim; use the installed command."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llms_experiments import _core
from llms_experiments.cli import main


def __getattr__(name: str):
    """Forward imported v0.2 symbols for one release cycle."""

    if name == "NvidiaAPIBackend":
        warnings.warn("NvidiaAPIBackend was renamed OpenAICompatibleBackend", DeprecationWarning, stacklevel=2)
        return _core.OpenAICompatibleBackend
    return getattr(_core, name)


def _forward(arguments: list[str]) -> list[str]:
    if not arguments:
        return arguments
    command = {"run-matrix": "run", "prepare-matrix": "prepare"}.get(arguments[0], arguments[0])
    forwarded = [command, *arguments[1:]]
    if "--config" in forwarded:
        index = forwarded.index("--config")
        config = forwarded[index + 1]
        del forwarded[index : index + 2]
        forwarded.insert(1, config)
    return forwarded


if __name__ == "__main__":
    warnings.warn(
        "experiment-cli/experiment_cli.py is deprecated and will be removed in v0.3; use `llms-experiments`.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main(_forward(sys.argv[1:])))
