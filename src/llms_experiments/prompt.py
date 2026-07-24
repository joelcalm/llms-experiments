"""Compatibility facade for prompt rendering and processor materialization."""

from .prompting import (
    build_requests,
    conversation,
    materialize_variant,
    prompt,
    prompt_group_id,
    prompt_values,
    read_asset,
    render,
    rendered_prompt,
    request_for_row,
    retry_values,
    system_prompt,
    variant_config_hash,
    variant_schema,
)

__all__ = [
    "build_requests",
    "conversation",
    "materialize_variant",
    "prompt",
    "prompt_group_id",
    "prompt_values",
    "read_asset",
    "render",
    "rendered_prompt",
    "request_for_row",
    "retry_values",
    "system_prompt",
    "variant_config_hash",
    "variant_schema",
]
