"""Candidate logprobs processing stage implementation."""

from __future__ import annotations

from dataclasses import replace

from ..base import ProcessingStage
from ..contracts import ProcessingError, ProcessingState, ProcessorContext, ResponseRequirements
from ..utils import aggregate_candidate_logprobs


class CandidateLogprobsStage(ProcessingStage):
    """Extracts first-token logprobabilities across candidate target labels."""

    type_name = "candidate_logprobs"
    required_fields = frozenset({"token_positions"})
    produced_fields = frozenset({"value", "candidate_scores"})

    def _candidates(self, context: ProcessorContext) -> tuple[str, ...]:
        """Resolve candidate label tuple from configuration or context source."""
        declared = self.config.get("candidates")
        source = self.config.get("candidates_from")
        if declared is not None and source is not None:
            raise ValueError("candidate_logprobs accepts candidates or candidates_from, not both")
        if declared is not None:
            candidates = tuple(str(item) for item in declared)
        elif source == "dataset_labels":
            candidates = context.dataset_labels
        elif source == "code_labels":
            candidates = tuple(str(item) for item in context.code_labels)
        else:
            raise ValueError("candidate_logprobs requires candidates or a supported candidates_from")
        if not candidates:
            raise ValueError("candidate_logprobs candidate set must not be empty")
        return candidates

    def requirements(self, context: ProcessorContext) -> ResponseRequirements:
        """Return response requirements for single-token logprob capture."""
        candidates = self._candidates(context)
        return ResponseRequirements(
            max_tokens=1,
            capture_logprobs=True,
            top_logprobs=min(20, len(candidates) + 5),
            one_token=True,
            candidates=candidates,
        )

    def process(self, state: ProcessingState, context: ProcessorContext) -> ProcessingState:
        """Extract candidate logprobabilities and populate candidate_scores in state."""
        candidates = self._candidates(context)
        if state.response.token_positions:
            scores = aggregate_candidate_logprobs(state.response.token_positions[0], candidates)
        elif state.response.candidate_scores is not None:
            scores = {
                candidate: float(state.response.candidate_scores.get(candidate, -float("inf")))
                for candidate in candidates
            }
        else:
            return state.fail(
                ProcessingError(
                    code="candidate_logprobs_missing",
                    message="backend returned no positional token logprobs",
                    stage=self.type_name,
                    category="processing",
                )
            )
        return replace(state, value={"candidates": scores}, candidate_scores=scores)
