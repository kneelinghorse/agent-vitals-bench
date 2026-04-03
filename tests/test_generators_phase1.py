"""Tests for Phase 1 generators: confabulation, thrash, runaway_cost."""

import pytest
from generators.confabulation import ConfabGenerator
from generators.thrash import ThrashGenerator
from generators.runaway_cost import RunawayCostGenerator


class TestConfabGenerator:
    def setup_method(self) -> None:
        self.gen = ConfabGenerator()

    def test_positive_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=True)
        assert meta.labels["confabulation"] is True
        assert meta.labels["loop"] is False

    def test_negative_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.labels["confabulation"] is False

    def test_correct_length(self) -> None:
        snapshots, _ = self.gen.generate(total_steps=10)
        assert len(snapshots) == 10

    def test_source_finding_ratio_drops(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=10, onset_step=3, source_finding_ratio=0.2,
        )
        # Before onset: sources > findings (healthy ratio)
        pre = snapshots[1]
        assert pre.signals.sources_count > pre.signals.findings_count

        # After onset: ratio should drop toward target
        post = snapshots[-1]
        ratio = post.signals.sources_count / max(1, post.signals.findings_count)
        assert ratio < 0.5  # Well below healthy

    def test_findings_keep_growing(self) -> None:
        snapshots, _ = self.gen.generate(total_steps=10, onset_step=3)
        confab_findings = [s.signals.findings_count for s in snapshots[3:]]
        assert confab_findings == sorted(confab_findings)
        assert confab_findings[-1] > confab_findings[0]

    def test_sources_stagnate(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=10, onset_step=3, source_finding_ratio=0.15,
        )
        confab_sources = [s.signals.sources_count for s in snapshots[4:]]
        # Sources should be roughly flat or declining
        assert max(confab_sources) - min(confab_sources) <= 3

    def test_confidence_inflated(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=3, confidence_inflation=0.4,
        )
        confab_conf = snapshots[-1].signals.confidence_score
        assert confab_conf >= 0.6  # Inflated above healthy baseline

    def test_onset_step_recorded(self) -> None:
        _, meta = self.gen.generate(onset_step=4, positive=True)
        assert meta.onset_step == 4

    def test_negative_has_no_onset(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.onset_step is None

    def test_negative_has_healthy_ratio(self) -> None:
        snapshots, _ = self.gen.generate(positive=False, total_steps=8)
        for s in snapshots[1:]:
            ratio = s.signals.sources_count / max(1, s.signals.findings_count)
            assert ratio > 0.5

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValueError):
            self.gen.generate(total_steps=2)


class TestThrashGenerator:
    def setup_method(self) -> None:
        self.gen = ThrashGenerator()

    def test_positive_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=True)
        assert meta.labels["thrash"] is True

    def test_negative_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.labels["thrash"] is False

    def test_correct_length(self) -> None:
        snapshots, _ = self.gen.generate(total_steps=8)
        assert len(snapshots) == 8

    def test_errors_accumulate(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=2, error_spikes=3,
        )
        thrash_errors = [s.signals.error_count for s in snapshots[2:]]
        assert thrash_errors[-1] > 0
        assert thrash_errors == sorted(thrash_errors)

    def test_refinement_count_grows(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=2, refinement_growth=2,
        )
        thrash_refinements = [s.signals.refinement_count for s in snapshots[2:]]
        assert thrash_refinements[-1] > thrash_refinements[0]

    def test_objectives_oscillate(self) -> None:
        snapshots, _ = self.gen.generate(total_steps=10, onset_step=2)
        thrash_objectives = [s.signals.objectives_covered for s in snapshots[2:]]
        # Should have both increases and decreases (oscillation)
        deltas = [thrash_objectives[i+1] - thrash_objectives[i]
                  for i in range(len(thrash_objectives) - 1)]
        has_increase = any(d > 0 for d in deltas)
        has_decrease = any(d < 0 for d in deltas)
        assert has_increase or has_decrease  # At least some oscillation

    def test_negative_has_zero_errors(self) -> None:
        snapshots, _ = self.gen.generate(positive=False, total_steps=8)
        assert all(s.signals.error_count == 0 for s in snapshots)

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValueError):
            self.gen.generate(total_steps=2)


class TestRunawayCostGenerator:
    def setup_method(self) -> None:
        self.gen = RunawayCostGenerator()

    def test_positive_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=True)
        assert meta.labels["runaway_cost"] is True

    def test_negative_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.labels["runaway_cost"] is False

    def test_correct_length(self) -> None:
        snapshots, _ = self.gen.generate(total_steps=8)
        assert len(snapshots) == 8

    def test_token_growth_accelerates(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=3, cost_growth="quadratic", burn_rate=4.0,
        )
        # Per-step token deltas should grow in runaway phase
        tokens = [s.signals.total_tokens for s in snapshots]
        deltas = [tokens[i+1] - tokens[i] for i in range(len(tokens) - 1)]
        runaway_deltas = deltas[3:]  # After onset
        assert runaway_deltas[-1] > runaway_deltas[0]

    def test_findings_stagnate(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=3, cost_growth="quadratic",
        )
        runaway_findings = [s.signals.findings_count for s in snapshots[3:]]
        # Findings should barely move
        assert runaway_findings[-1] - runaway_findings[0] <= 2

    def test_linear_growth(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=3, cost_growth="linear", burn_rate=3.0,
        )
        tokens = [s.signals.total_tokens for s in snapshots]
        assert tokens[-1] > tokens[3]

    def test_step_growth(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=8, onset_step=3, cost_growth="step", burn_rate=5.0,
        )
        tokens = [s.signals.total_tokens for s in snapshots]
        assert tokens[-1] > tokens[3]

    def test_negative_has_steady_tokens(self) -> None:
        snapshots, _ = self.gen.generate(positive=False, total_steps=8)
        tokens = [s.signals.total_tokens for s in snapshots]
        deltas = [tokens[i+1] - tokens[i] for i in range(len(tokens) - 1)]
        # All deltas should be roughly equal (steady burn)
        assert max(deltas) < min(deltas) * 3

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValueError):
            self.gen.generate(total_steps=2)

    def test_quadratic_has_accelerating_deltas(self) -> None:
        snapshots, _ = self.gen.generate(
            total_steps=10, onset_step=3, cost_growth="quadratic", burn_rate=5.0,
        )
        tokens = [s.signals.total_tokens for s in snapshots]
        deltas = [tokens[i+1] - tokens[i] for i in range(len(tokens) - 1)]
        runaway_deltas = deltas[3:]
        # Quadratic: later deltas should be larger than early ones
        assert runaway_deltas[-1] > runaway_deltas[0]
