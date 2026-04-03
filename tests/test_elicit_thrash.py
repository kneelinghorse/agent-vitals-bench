"""Unit tests for elicitation/elicit_thrash.py — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

from elicitation.elicit_thrash import (
    COHERENT_TASKS,
    CONFLICT_SCENARIOS,
    ElicitationRun,
    ThrashStepResult,
    _objectives_oscillate,
    elicit_thrash,
)
from elicitation.providers import GenerateResult, Provider


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


def _make_mock_provider(
    content: str = "Here is my response with a different approach.",
    name: str = "test-provider",
    model: str = "test-model",
) -> Provider:
    mock = MagicMock(spec=Provider)
    mock.name = name
    mock._model = model

    def _generate(prompt: str, **kwargs: object) -> GenerateResult:
        return GenerateResult(
            content=content,
            prompt_tokens=80,
            completion_tokens=150,
            total_tokens=230,
            model=model,
            provider=name,
        )

    mock.generate = _generate
    return mock


# ---------------------------------------------------------------------------
# _objectives_oscillate
# ---------------------------------------------------------------------------


class TestObjectivesOscillate:
    def test_oscillation(self) -> None:
        assert _objectives_oscillate([1, 3, 2]) is True

    def test_monotonic_increase(self) -> None:
        assert _objectives_oscillate([1, 2, 3, 4]) is False

    def test_monotonic_decrease(self) -> None:
        assert _objectives_oscillate([4, 3, 2, 1]) is False

    def test_too_short(self) -> None:
        assert _objectives_oscillate([1, 2]) is False

    def test_complex_oscillation(self) -> None:
        assert _objectives_oscillate([3, 5, 2, 4, 1]) is True

    def test_flat(self) -> None:
        assert _objectives_oscillate([3, 3, 3, 3]) is False


# ---------------------------------------------------------------------------
# Positive elicitation
# ---------------------------------------------------------------------------


class TestPositiveElicitation:
    def test_produces_positive_trace(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(
            provider, positive=True, total_steps=8, trace_id="test-thrash-pos"
        )

        assert isinstance(run, ElicitationRun)
        assert run.positive is True
        assert run.metadata.labels["thrash"] is True
        assert run.metadata.tier == "elicited"
        assert run.metadata.generator == "ThrashElicitor"
        assert len(run.snapshots) == 8
        assert len(run.steps) == 8

    def test_has_contradictory_steps(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=6)

        contradictions = [s for s in run.steps if s.is_contradiction]
        assert len(contradictions) > 0

    def test_error_spikes_present(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=8)

        error_steps = [s for s in run.snapshots if s.signals.error_count > 0]
        assert len(error_steps) > 0

    def test_refinements_climb(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=8)

        last = run.snapshots[-1]
        assert last.signals.refinement_count >= 3

    def test_objectives_oscillate_in_trace(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=8)

        obj_seq = [s.signals.objectives_covered for s in run.snapshots]
        assert _objectives_oscillate(obj_seq)

    def test_onset_step_recorded(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=8)

        assert run.metadata.onset_step is not None

    def test_token_accumulation(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=4)

        # Each step is 230 tokens
        assert run.snapshots[-1].signals.total_tokens == 230 * 4


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


class TestNegativeElicitation:
    def test_produces_negative_trace(self) -> None:
        provider = _make_mock_provider(content="Here is a clear answer.")
        run = elicit_thrash(
            provider, positive=False, total_steps=6, trace_id="test-thrash-neg"
        )

        assert run.metadata.labels["thrash"] is False
        assert run.metadata.tier == "elicited"

    def test_no_contradictions(self) -> None:
        provider = _make_mock_provider(content="Straightforward answer.")
        run = elicit_thrash(provider, positive=False, total_steps=4)

        assert all(not s.is_contradiction for s in run.steps)

    def test_no_errors(self) -> None:
        provider = _make_mock_provider(content="Clean response.")
        run = elicit_thrash(provider, positive=False, total_steps=4)

        assert all(s.signals.error_count == 0 for s in run.snapshots)

    def test_monotonic_objectives(self) -> None:
        provider = _make_mock_provider(content="Answer.")
        run = elicit_thrash(provider, positive=False, total_steps=5)

        obj_seq = [s.signals.objectives_covered for s in run.snapshots]
        # Should be monotonically non-decreasing
        for i in range(1, len(obj_seq)):
            assert obj_seq[i] >= obj_seq[i - 1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_trace_id_auto_generated(self) -> None:
        provider = _make_mock_provider()

        run_pos = elicit_thrash(provider, positive=True, total_steps=4)
        assert run_pos.trace_id.startswith("thrash-elic-pos-")

        run_neg = elicit_thrash(provider, positive=False, total_steps=4)
        assert run_neg.trace_id.startswith("thrash-elic-neg-")

    def test_snapshots_valid(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=6)

        for i, snap in enumerate(run.snapshots):
            assert snap.loop_index == i
            assert snap.mission_id == f"bench-{run.trace_id}"
            assert 0.0 <= snap.signals.coverage_score <= 1.0
            assert 0.0 <= snap.signals.confidence_score <= 1.0

    def test_step_details_captured(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=4)

        for step in run.steps:
            assert isinstance(step, ThrashStepResult)
            assert len(step.prompt) > 0
            assert len(step.response) > 0
            assert step.total_tokens > 0

    def test_confidence_scoring(self) -> None:
        provider = _make_mock_provider()
        run = elicit_thrash(provider, positive=True, total_steps=8)

        assert run.metadata.confidence in (0.8, 0.9)


# ---------------------------------------------------------------------------
# Prompt banks
# ---------------------------------------------------------------------------


class TestPromptBanks:
    def test_conflict_scenarios_populated(self) -> None:
        assert len(CONFLICT_SCENARIOS) >= 5

    def test_coherent_tasks_populated(self) -> None:
        assert len(COHERENT_TASKS) >= 5

    def test_scenarios_have_all_fields(self) -> None:
        for s in CONFLICT_SCENARIOS:
            assert len(s.initial) > 0
            assert len(s.contradiction) > 0
            assert len(s.topic) > 0
