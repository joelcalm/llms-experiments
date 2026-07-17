"""Variants may carry a system turn, and every request path must send it.

The runner used to send a single `user` message from every backend, which is
why slurm/batch_infer_simple.py (system + user) could not be expressed as a
config. A system prompt is instructions, not data, so it is rendered once per
variant rather than once per row.
"""

from __future__ import annotations

import json
from pathlib import Path

import experiment_cli as cli
import pytest
from conftest import REPO_ROOT

SMOKE = REPO_ROOT / "experiments" / "local_all_modes_smoke.yaml"


@pytest.fixture
def config(tmp_path: Path):
    def _load(system: str | list[str] | None = None, body: str = "You are a classifier. Labels: {{labels}}."):
        loaded = cli.load_config(SMOKE, [f"output.directory={tmp_path / 'out'}"], check_files=True)
        loaded["variants"] = [v for v in loaded["variants"] if v["id"] == "single_label_json"]
        if system is not None:
            path = tmp_path / "system.md"
            path.write_text(body, encoding="utf-8")
            loaded["variants"][0]["system_prompt"] = str(path) if system == "str" else [str(path)]
        return loaded

    return _load


def test_no_system_prompt_means_no_system_turn(config) -> None:
    """The default must stay exactly as it was: a single user message."""
    loaded = config()
    variant = cli.materialize_variant(loaded, loaded["variants"][0])

    assert cli.system_prompt(loaded, variant, None) is None
    assert cli.conversation(None, "hello") == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("declared", ["str", "list"])
def test_system_prompt_accepts_a_string_or_a_list(config, declared: str) -> None:
    loaded = config(system=declared)
    variant = cli.materialize_variant(loaded, loaded["variants"][0])

    rendered = cli.system_prompt(loaded, variant, None)

    assert rendered is not None
    assert "alpha, beta, gamma" in rendered, "the system turn renders the same placeholders as any prompt"


def test_system_prompt_reaches_the_batch_request(config) -> None:
    loaded = config(system="list")
    variant = cli.materialize_variant(loaded, loaded["variants"][0])
    row = {"id": "r1", "text": "some text", "_source_position": 0}

    request = cli.request_for_row(loaded, variant, row, cli.variant_schema(loaded, variant))

    roles = [message["role"] for message in request["body"]["messages"]]
    assert roles == ["system", "user"], "run-batch must carry the system turn"
    assert "You are a classifier" in request["body"]["messages"][0]["content"]
    assert "some text" in request["body"]["messages"][1]["content"]


def test_system_prompt_rejects_row_placeholders(config) -> None:
    """{{text}} in a system prompt would render empty for every row, silently."""
    loaded = config(system="list", body="Classify: {{text}}")
    variant = cli.materialize_variant(loaded, loaded["variants"][0])

    with pytest.raises(ValueError, match="system_prompt may not use"):
        cli.system_prompt(loaded, variant, None)


def test_system_prompt_changes_the_config_hash_and_group_id(config, tmp_path: Path) -> None:
    """Editing the system prompt must invalidate resume, like any other asset."""
    plain = config()
    plain_variant = cli.materialize_variant(plain, plain["variants"][0])
    withsystem = config(system="list")
    system_variant = cli.materialize_variant(withsystem, withsystem["variants"][0])

    assert cli.variant_config_hash(plain, plain_variant) != cli.variant_config_hash(withsystem, system_variant)
    assert cli.prompt_group_id(plain, plain_variant, None) != cli.prompt_group_id(withsystem, system_variant, None)

    # Same variant, edited system prompt -> different identity again.
    (tmp_path / "system.md").write_text("A different instruction entirely.", encoding="utf-8")
    cli._read_asset.cache_clear()
    assert cli.prompt_group_id(withsystem, system_variant, None) != cli.prompt_group_id(plain, plain_variant, None)


def test_missing_system_prompt_file_fails_validation(config, tmp_path: Path) -> None:
    loaded = config()
    loaded["variants"][0]["system_prompt"] = [str(tmp_path / "absent.md")]

    with pytest.raises(ValueError):
        cli.validate_config(loaded, check_files=True)


def test_run_sends_the_system_turn_to_the_backend(config) -> None:
    """End to end: the engine must hand the backend the rendered system turn."""
    seen: list[dict] = []

    class CapturingBackend:
        def generate(self, prompts: list[str], variant: dict) -> list[cli.Response]:
            seen.append(variant)
            return [cli.Response(json.dumps({"label": "alpha"}), 3, None, None) for _ in prompts]

        def close(self) -> None:
            return None

    loaded = config(system="list")
    loaded["logging"] = {"file": "/dev/null", "events": "/dev/null"}
    cli.run(loaded, CapturingBackend(), row_limit=2)

    assert seen, "the backend was never called"
    assert "You are a classifier" in (seen[0].get("_system") or "")
