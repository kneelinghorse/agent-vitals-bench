"""Unit tests for elicitation/elicit_runaway_cost.py — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

from elicitation.elicit_runaway_cost import (
    BURN_RATE_MULTIPLIER,
    CONSTRAINED_TASKS,
    ESCALATING_TASKS,
    CostStepResult,
    ElicitationRun,
    _compute_baseline,
    _find_burn_onset,
    _tokens_increasing,
    elicit_runaway_cost,
)
from elicitation.providers import GenerateResult, Provider


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


def _make_escalating_provider(
    base_tokens: int = 200,
    growth_factor: float = 2.0,
    name: str = "test-provider",
    model: str = "test-model",
) -> Provider:
    """Provider that returns escalating token counts per call."""
    mock = MagicMock(spec=Provider)
    mock.name = name
    mock._model = model

    call_count = 0

    def _generate(prompt: str, **kwargs: object) -> GenerateResult:
        nonlocal call_count
        # First 3 calls: baseline tokens. After that: growing.
        if call_count < 3:
            tokens = base_tokens
        else:
            tokens = int(base_tokens * growth_factor ** (call_count - 2))
        call_count += 1
        p_tokens = int(tokens * 0.4)
        c_tokens = tokens - p_tokens
        return GenerateResult(
            content="Response " * min(50, tokens // 5),
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=tokens,
            model=model,
            provider=name,
        )

    mock.generate = _generate
    return mock


def _make_flat_provider(
    tokens: int = 200,
    name: str = "test-provider",
    model: str = "test-model",
) -> Provider:
    """Provider that returns constant token counts."""
    mock = MagicMock(spec=Provider)
    mock.name = name
    mock._model = model

    def _generate(prompt: str, **kwargs: object) -> GenerateResult:
        return GenerateResult(
            content="Short response.",
            prompt_tokens=int(tokens * 0.4),
            completion_tokens=tokens - int(tokens * 0.4),
            total_tokens=tokens,
            model=model,
            provider=name,
        )

    mock.generate = _generate
    return mock


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestComputeBaseline:
    def test_normal(self) -> None:
        assert _compute_baseline([100, 200, 300]) == 200.0

    def test_fewer_than_n(self) -> None:
        assert _compute_baseline([100, 200], n_baseline=3) == 150.0

    def test_empty(self) -> None:
        assert _compute_baseline([]) == 0.0

    def test_uses_first_n(self) -> None:
        assert _compute_baseline([100, 200, 300, 10000]) == 200.0


class TestFindBurnOnset:
    def test_no_onset_flat(self) -> None:
        assert _find_burn_onset([100, 100, 100, 100, 100]) is None

    def test_onset_at_step_3(self) -> None:
        # Baseline = avg(100,100,100) = 100, threshold = 300
        assert _find_burn_onset([100, 100, 100, 400, 800]) == 3

    def test_too_short(self) -> None:
        assert _find_burn_onset([100, 200, 300]) is None

    def test_gradual_growth_no_onset(self) -> None:
        # Growth but never exceeds 3x baseline
        assert _find_burn_onset([100, 100, 100, 200, 250]) is None


class TestTokensIncreasing:
    def test_increasing(self) -> None:
        assert _tokens_increasing([100, 200, 300, 400]) is True

    def test_flat(self) -> None:
        assert _tokens_increasing([100, 100, 100, 100]) is False

    def test_decreasing(self) -> None:
        assert _tokens_increasing([400, 300, 200, 100]) is False

    def test_too_short(self) -> None:
        assert _tokens_increasing([100, 200]) is False


# ---------------------------------------------------------------------------
# Positive elicitation
# ---------------------------------------------------------------------------


class TestPositiveElicitation:
    def test_produces_positive_trace(self) -> None:
        provider = _make_escalating_provider(base_tokens=200, growth_factor=2.5)
        run = elicit_runaway_cost(
            provider, positive=True, total_steps=8, trace_id="test-runaway-pos"
        )

        assert isinstance(run, ElicitationRun)
        assert run.positive is True
        assert run.metadata.labels["runaway_cost"] is True
        assert run.metadata.tier == "elicited"
        assert run.metadata.generator == "RunawayCostElicitor"
        assert len(run.snapshots) == 8

    def test_onset_step_recorded(self) -> None:
        provider = _make_escalating_provider(base_tokens=200, growth_factor=2.5)
        run = elicit_runaway_cost(provider, positive=True, total_steps=8)

        assert run.metadata.onset_step is not None
        assert run.metadata.onset_step >= 3  # After baseline window

    def test_peak_ratio_exceeds_threshold(self) -> None:
        provider = _make_escalating_provider(base_tokens=200, growth_factor=2.5)
        run = elicit_runaway_cost(provider, positive=True, total_steps=8)

        assert run.metadata.params["peak_ratio"] > BURN_RATE_MULTIPLIER

    def test_token_growth_in_snapshots(self) -> None:
        provider = _make_escalating_provider(base_tokens=200, growth_factor=2.0)
        run = elicit_runaway_cost(provider, positive=True, total_steps=6)

        # Cumulative tokens should always increase
        cumulatives = [s.signals.total_tokens for s in run.snapshots]
        for i in range(1, len(cumulatives)):
            assert cumulatives[i] > cumulatives[i - 1]

    def test_step_details_captured(self) -> None:
        provider = _make_escalating_provider()
        run = elicit_runaway_cost(provider, positive=True, total_steps=4)

        for step in run.steps:
            assert isinstance(step, CostStepResult)
            assert len(step.prompt) > 0
            assert step.total_tokens > 0


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


class TestNegativeElicitation:
    def test_produces_negative_trace(self) -> None:
        provider = _make_flat_provider(tokens=200)
        run = elicit_runaway_cost(
            provider, positive=False, total_steps=6, trace_id="test-runaway-neg"
        )

        assert run.metadata.labels["runaway_cost"] is False
        assert run.metadata.tier == "elicited"

    def test_flat_tokens(self) -> None:
        provider = _make_flat_provider(tokens=200)
        run = elicit_runaway_cost(provider, positive=False, total_steps=6)

        step_tokens = [s.total_tokens for s in run.steps]
        # All steps should have same token count
        assert all(t == step_tokens[0] for t in step_tokens)

    def test_no_onset(self) -> None:
        provider = _make_flat_provider(tokens=200)
        run = elicit_runaway_cost(provider, positive=False, total_steps=6)

        assert run.metadata.onset_step is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_trace_id_auto_generated(self) -> None:
        provider = _make_flat_provider()

        run_pos = elicit_runaway_cost(provider, positive=True, total_steps=4)
        assert run_pos.trace_id.startswith("runaway-elic-pos-")

        run_neg = elicit_runaway_cost(provider, positive=False, total_steps=4)
        assert run_neg.trace_id.startswith("runaway-elic-neg-")

    def test_snapshots_valid(self) -> None:
        provider = _make_escalating_provider()
        run = elicit_runaway_cost(provider, positive=True, total_steps=6)

        for i, snap in enumerate(run.snapshots):
            assert snap.loop_index == i
            assert snap.mission_id == f"bench-{run.trace_id}"
            assert 0.0 <= snap.signals.coverage_score <= 1.0
            assert 0.0 <= snap.signals.confidence_score <= 1.0
            assert snap.signals.total_tokens > 0

    def test_confidence_scoring(self) -> None:
        provider = _make_escalating_provider(base_tokens=200, growth_factor=3.0)
        run = elicit_runaway_cost(provider, positive=True, total_steps=8)

        assert run.metadata.confidence in (0.8, 0.9)

    def test_params_recorded(self) -> None:
        provider = _make_escalating_provider()
        run = elicit_runaway_cost(provider, positive=True, total_steps=6)

        params = run.metadata.params
        assert "baseline_tokens" in params
        assert "peak_tokens" in params
        assert "peak_ratio" in params
        assert "is_increasing" in params


# ---------------------------------------------------------------------------
# Task banks
# ---------------------------------------------------------------------------


class TestTaskBanks:
    def test_escalating_tasks_populated(self) -> None:
        assert len(ESCALATING_TASKS) >= 5

    def test_constrained_tasks_populated(self) -> None:
        assert len(CONSTRAINED_TASKS) >= 5

    def test_escalating_tasks_have_followups(self) -> None:
        for task in ESCALATING_TASKS:
            assert len(task.followups) >= 3
            assert len(task.initial) > 0
            assert len(task.topic) > 0
