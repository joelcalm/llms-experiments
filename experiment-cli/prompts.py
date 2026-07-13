"""Generic Markdown prompt composition with deliberately limited substitutions."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"{{\s*(text|row_id|output_schema|raw_response|validation_errors)\s*}}")


def render(template: str, values: dict[str, Any]) -> str:
    return _TOKEN.sub(lambda match: str(values.get(match.group(1), "")), template)


def compose(root: Path, paths: list[str], values: dict[str, Any]) -> str:
    return "\n\n".join(render((root / path).read_text(encoding="utf-8").strip(), values) for path in paths)

