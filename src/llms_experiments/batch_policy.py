"""Policies for recovering from an inference batch-capacity failure."""

from __future__ import annotations

from collections.abc import Sequence


def conservative_retry_batch(safe_candidates: Sequence[int]) -> int | None:
    """Return a robust retry size after a backend dies during probing.

    The largest successful warm-up probe can be optimistic: a later lane row
    may have a longer prompt or exercise a higher activation peak. Retry at
    the preceding successful candidate when one exists, or at half the sole
    candidate. Return ``None`` when no positive candidate was accepted.
    """

    ordered = sorted({int(candidate) for candidate in safe_candidates if int(candidate) > 0})
    if not ordered:
        return None
    if len(ordered) > 1:
        return ordered[-2]
    return max(1, ordered[-1] // 2)
