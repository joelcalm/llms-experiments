"""Regression checks for model-specific quantised worker controls."""

from __future__ import annotations

from pathlib import Path


def test_quantized_worker_exposes_json_compatibility_controls() -> None:
    """The worker must forward Muse-compatible generation controls."""
    source = (
        Path(__file__).parents[1] / "configs" / "protoethos" / "condor" / "run_quantized_full_matrix_worker.sh"
    ).read_text(encoding="utf-8")

    assert 'STRUCTURED_OUTPUTS="${STRUCTURED_OUTPUTS:-}"' in source
    assert 'GENERATION_CONFIG="${GENERATION_CONFIG:-}"' in source
    assert 'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"' in source
    assert 'CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES:-}"' in source
    assert 'BATCH_REFINE_AFTER_FAILURE="${BATCH_REFINE_AFTER_FAILURE:-false}"' in source
    assert 'ENV_PYTHON="${RUNNER_PYTHON:-$PROJECT_DIR/.venv/bin/python}"' in source
    assert '"$ENV_PYTHON" -u "$CODE_DIR/experiment-cli/experiment_cli.py" run-matrix' in source
    assert 'MODEL_BACKEND="${MODEL_BACKEND:-local_vllm}"' in source
    assert '--set "model.backend=$MODEL_BACKEND"' in source
    assert '--set "output.checkpoint_name=checkpoint.${VARIANT_LANE}.json"' in source
    assert "trap on_interrupt TERM INT" in source
    assert 'append_override "${STRUCTURED_OUTPUTS:-}" model.structured_outputs' in source
    assert 'append_override "${GENERATION_CONFIG:-}" model.generation_config' in source
    assert 'append_override "${KV_CACHE_DTYPE:-}" model.kv_cache_dtype' in source
    assert 'append_override "${CALCULATE_KV_SCALES:-}" model.calculate_kv_scales' in source
    assert '--set "batch.refine_after_failure=$BATCH_REFINE_AFTER_FAILURE"' in source
    assert "conservative_retry_batch" in source
    assert "Restrict a resumed matrix job to lanes" in source
    assert 'checkpoint.get("state") == "completed"' in source
    assert "single_label_json.parquet" in source
    assert 'DATASETS="$missing_datasets"' in source
    assert 'RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-${RESULTS_ROOT%/*}/logs}"' in source


def test_quantized_worker_uses_job_local_runtime_caches() -> None:
    """Disposable compiler caches must not consume the shared Lustre quota."""
    source = (
        Path(__file__).parents[1] / "configs" / "protoethos" / "condor" / "run_quantized_full_matrix_worker.sh"
    ).read_text(encoding="utf-8")

    assert 'runtime_cache_root="${JOB_CACHE_ROOT:-${SHARED_CACHE_ROOT:-$RUN_ROOT/shared_cache}}"' in source
    assert 'export TORCHINDUCTOR_CACHE_DIR="$runtime_cache_root/torchinductor"' in source
    assert 'export TRITON_CACHE_DIR="$runtime_cache_root/triton"' in source
    assert 'export XDG_CACHE_HOME="$runtime_cache_root/xdg_cache"' in source


def test_quantized_worker_exposes_async_engine_and_hybrid_cache_controls() -> None:
    """The continuous dispatcher needs reproducible engine and queue settings."""
    source = (
        Path(__file__).parents[1] / "configs" / "protoethos" / "condor" / "run_quantized_full_matrix_worker.sh"
    ).read_text(encoding="utf-8")

    for setting in (
        "STREAMING_DISPATCH",
        "STREAM_MAX_IN_FLIGHT",
        "STREAM_CHECKPOINT_ROWS",
        "MAMBA_CACHE_DTYPE",
        "MAMBA_SSM_CACHE_DTYPE",
        "PERFORMANCE_MODE",
        "OPTIMIZATION_LEVEL",
        "VLLM_USE_FASTOKENS",
    ):
        assert setting in source
