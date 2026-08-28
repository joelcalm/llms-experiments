from llms_experiments.batch_policy import conservative_retry_batch


def test_retry_uses_preceding_probe_after_engine_death() -> None:
    """Do not retry the largest optimistic probe after a later OOM."""
    assert conservative_retry_batch([256, 512]) == 256


def test_retry_halves_a_single_accepted_probe() -> None:
    """Leave headroom when the only accepted candidate may be optimistic."""
    assert conservative_retry_batch([257]) == 128


def test_retry_returns_none_without_positive_candidates() -> None:
    """An empty result must not cause the worker to retry blindly."""
    assert conservative_retry_batch([0, -1]) is None
