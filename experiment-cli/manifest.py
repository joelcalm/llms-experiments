"""Manifest writer kept separate for auditability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_manifest(run_dir: Path, config: dict[str, Any], manifest: dict[str, Any]) -> None:
    effective = {key: value for key, value in config.items() if not key.startswith("_")}
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

