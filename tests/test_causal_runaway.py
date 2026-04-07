"""Tests for the causal runaway-cost prototype."""

from __future__ import annotations

from generators.runaway_cost import RunawayCostGenerator
from prototypes.causal_runaway import (
    CausalRunawayConfig,
    detect_causal_runaway_cost,
    evaluate_causal_runaway_corpus,
    score_runaway_windows,
    sweep_cost_growth_multiplier,
)


class TestCausalRunawayPrototype:
    def setup_method(self) -> None:
        self.generator = RunawayCostGenerator()
        self.config = CausalRunawayConfig()

    def test_healthy_trace_does_not_trigger(self) -> None:
        snapshots, _ = self.generator.generate(positive=False, total_steps=8)

        result = detect_causal_runaway_cost(snapshots, self.config)

        assert result.detected is False
        assert result.trigger is None
        assert result.confidence < 0.5

    def test_classic_runaway_trace_triggers(self) -> None:
        snapshots, _ = self.generator.generate(
            positive=True,
            total_steps=8,
            onset_step=3,
            cost_growth="quadratic",
            burn_rate=4.0,
        )

        result = detect_causal_runaway_cost(snapshots, self.config)

        assert result.detected is True
        assert result.trigger == "cost_output_decoupling"
        assert result.cost_growth >= self.config.cost_growth_multiplier
        assert result.peak_cost >= self.config.absolute_cost_floor

    def test_gradual_onset_runaway_trace_triggers(self) -> None:
        snapshots, _ = self.generator.generate(
            positive=True,
            total_steps=10,
            onset_step=4,
            cost_growth="linear",
            burn_rate=4.0,
        )

        result = detect_causal_runaway_cost(snapshots, self.config)

        assert result.detected is True
        assert result.onset_step is not None
        assert result.worst_correlation <= self.config.correlation_ceiling

    def test_short_trace_returns_insufficient_history(self) -> None:
        snapshots, _ = self.generator.generate(positive=False, total_steps=6)

        result = detect_causal_runaway_cost(snapshots[:3], self.config)

        assert result.detected is False
        assert result.trigger == "insufficient_history"

    def test_missing_optional_fields_do_not_break_scoring(self) -> None:
        snapshots, _ = self.generator.generate(positive=True, total_steps=8, onset_step=3)
        stripped = tuple(
            snapshot.model_copy(
                update={
                    "loop_trigger": None,
                    "stuck_trigger": None,
                    "detector_priority": None,
                }
            )
            for snapshot in snapshots
        )

        result = detect_causal_runaway_cost(stripped, self.config)
        scores = score_runaway_windows(stripped, self.config)

        assert result.detected is True
        assert len(scores) > 0

    def test_runaway_corpus_beats_handcrafted_baseline(self) -> None:
        evaluation = evaluate_causal_runaway_corpus(
            corpus_version="v1",
            config=self.config,
            min_confidence=0.8,
        )

        assert evaluation.trace_count > 0
        assert evaluation.metric.f1 >= 0.941

    def test_multiplier_safe_range_is_wide(self) -> None:
        sweep = sweep_cost_growth_multiplier(
            corpus_version="v1",
            config=self.config,
            min_confidence=0.8,
        )

        assert sweep.passing_min is not None
        assert sweep.passing_max is not None
        assert sweep.passing_width >= 1.0
