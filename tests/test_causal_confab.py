"""Tests for the causal confabulation prototype."""

from __future__ import annotations

from generators.confabulation import ConfabGenerator
from prototypes.causal_confab import (
    CausalConfabConfig,
    detect_causal_confabulation,
    evaluate_causal_confab_corpus,
    score_causal_windows,
)


class TestCausalConfabPrototype:
    def setup_method(self) -> None:
        self.generator = ConfabGenerator()
        self.config = CausalConfabConfig()

    def test_healthy_trace_does_not_trigger(self) -> None:
        snapshots, _ = self.generator.generate(positive=False, total_steps=8)

        result = detect_causal_confabulation(snapshots, self.config)

        assert result.detected is False
        assert result.trigger is None
        assert result.confidence < 0.5

    def test_classic_confab_trace_triggers(self) -> None:
        snapshots, _ = self.generator.generate(
            positive=True,
            total_steps=8,
            onset_step=2,
            source_finding_ratio=0.15,
        )

        result = detect_causal_confabulation(snapshots, self.config)

        assert result.detected is True
        assert result.confidence >= 0.6
        assert result.final_ratio <= 0.3
        assert result.onset_step is not None

    def test_delayed_onset_confab_trace_uses_structural_break_trigger(self) -> None:
        snapshots, _ = self.generator.generate(
            positive=True,
            total_steps=10,
            onset_step=4,
            source_finding_ratio=0.15,
            pattern="delayed_sharp",
        )

        result = detect_causal_confabulation(snapshots, self.config)

        assert result.detected is True
        assert result.trigger == "causal_link_break"
        assert result.baseline_strength > result.weakest_strength
        assert result.structural_drop >= self.config.structural_drop_threshold

    def test_short_trace_returns_insufficient_history(self) -> None:
        snapshots, _ = self.generator.generate(positive=False, total_steps=6)

        result = detect_causal_confabulation(snapshots[:3], self.config)

        assert result.detected is False
        assert result.trigger == "insufficient_history"
        assert result.onset_step is None

    def test_missing_optional_fields_do_not_break_scoring(self) -> None:
        snapshots, _ = self.generator.generate(positive=True, total_steps=8, onset_step=3)
        stripped = tuple(
            snapshot.model_copy(
                update={
                    "source_finding_ratio": None,
                    "verified_source_ratio": None,
                    "confabulation_trigger": None,
                }
            )
            for snapshot in snapshots
        )

        result = detect_causal_confabulation(stripped, self.config)
        scores = score_causal_windows(stripped, self.config)

        assert result.detected is True
        assert len(scores) > 0

    def test_synthetic_corpus_beats_gate_baseline(self) -> None:
        evaluation = evaluate_causal_confab_corpus(
            corpus_version="v1",
            config=self.config,
            min_confidence=0.8,
            tier="synthetic",
        )

        assert evaluation.trace_count > 0
        assert evaluation.metric.precision >= 0.94
        assert evaluation.metric.recall >= 0.94
        assert evaluation.metric.f1 >= 0.94
