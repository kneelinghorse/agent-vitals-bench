"""Tests for synthetic trace generators."""

import pytest
from generators.loop import LoopGenerator
from generators.stuck import StuckGenerator


class TestLoopGenerator:
    def setup_method(self) -> None:
        self.gen = LoopGenerator()

    def test_positive_trace_has_correct_length(self) -> None:
        snapshots, meta = self.gen.generate(total_steps=10, loop_start=3, loop_length=4)
        assert len(snapshots) == 10

    def test_positive_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=True)
        assert meta.labels["loop"] is True
        assert meta.labels["stuck"] is False

    def test_negative_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.labels["loop"] is False

    def test_onset_step_recorded(self) -> None:
        _, meta = self.gen.generate(loop_start=5, positive=True)
        assert meta.onset_step == 5

    def test_negative_has_no_onset(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.onset_step is None

    def test_findings_plateau_in_positive(self) -> None:
        snapshots, _ = self.gen.generate(
            loop_start=3,
            loop_length=5,
            total_steps=10,
            pattern="exact",
        )
        # After loop_start, findings should freeze
        plateau_findings = [s.signals.findings_count for s in snapshots[3:]]
        assert all(f == plateau_findings[0] for f in plateau_findings)

    def test_tokens_keep_growing_in_loop(self) -> None:
        snapshots, _ = self.gen.generate(
            loop_start=3,
            loop_length=5,
            total_steps=10,
        )
        loop_tokens = [s.signals.total_tokens for s in snapshots[3:]]
        assert loop_tokens == sorted(loop_tokens)
        assert loop_tokens[-1] > loop_tokens[0]

    def test_negative_trace_has_growing_findings(self) -> None:
        snapshots, _ = self.gen.generate(positive=False, total_steps=8)
        findings = [s.signals.findings_count for s in snapshots]
        assert findings == sorted(findings)
        assert findings[-1] > findings[0]

    def test_manifest_entry_format(self) -> None:
        _, meta = self.gen.generate(trace_id="test-001")
        entry = meta.as_manifest_entry("traces/loop/positive/test-001.json")
        assert entry["trace_id"] == "test-001"
        assert entry["tier"] == "synthetic"
        assert entry["metadata"]["confidence"] == 1.0
        assert entry["metadata"]["generator"] == "LoopGenerator"

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be >= 3"):
            self.gen.generate(total_steps=2)

    def test_semantic_pattern_has_variation(self) -> None:
        snapshots, _ = self.gen.generate(
            loop_start=3,
            loop_length=6,
            total_steps=10,
            pattern="semantic",
        )
        loop_findings = [s.signals.findings_count for s in snapshots[3:]]
        # Semantic pattern should have some variation (not all identical)
        assert len(set(loop_findings)) > 1

    def test_exact_pattern_is_frozen(self) -> None:
        snapshots, _ = self.gen.generate(
            loop_start=3,
            loop_length=5,
            total_steps=10,
            pattern="exact",
        )
        loop_findings = [s.signals.findings_count for s in snapshots[3:]]
        assert len(set(loop_findings)) == 1


class TestStuckGenerator:
    def setup_method(self) -> None:
        self.gen = StuckGenerator()

    def test_positive_trace_has_correct_length(self) -> None:
        snapshots, meta = self.gen.generate(step_count=8)
        assert len(snapshots) == 8

    def test_positive_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=True)
        assert meta.labels["stuck"] is True
        assert meta.labels["loop"] is False

    def test_negative_trace_labels(self) -> None:
        _, meta = self.gen.generate(positive=False)
        assert meta.labels["stuck"] is False

    def test_dm_value_in_stuck_phase(self) -> None:
        snapshots, _ = self.gen.generate(stuck_start=3, dm_value=0.05, step_count=8)
        for s in snapshots[3:]:
            assert s.metrics.dm_coverage == pytest.approx(0.05, abs=0.01)

    def test_cv_value_in_stuck_phase(self) -> None:
        snapshots, _ = self.gen.generate(stuck_start=3, cv_value=0.1, step_count=8)
        for s in snapshots[3:]:
            assert s.metrics.cv_coverage == pytest.approx(0.1, abs=0.01)

    def test_tokens_burn_during_stuck(self) -> None:
        snapshots, _ = self.gen.generate(stuck_start=3, step_count=8)
        stuck_tokens = [s.signals.total_tokens for s in snapshots[3:]]
        assert stuck_tokens == sorted(stuck_tokens)
        assert stuck_tokens[-1] > stuck_tokens[0]

    def test_findings_near_frozen_during_stuck(self) -> None:
        snapshots, _ = self.gen.generate(stuck_start=3, step_count=8)
        stuck_findings = [s.signals.findings_count for s in snapshots[3:]]
        # Findings grow very slowly (//3 steps) to avoid loop false positives
        assert stuck_findings[-1] - stuck_findings[0] <= 2
        assert stuck_findings == sorted(stuck_findings)

    def test_convergence_delta_zero_when_stuck(self) -> None:
        snapshots, _ = self.gen.generate(stuck_start=3, step_count=8)
        for s in snapshots[3:]:
            assert s.signals.convergence_delta == 0.0

    def test_negative_has_progress(self) -> None:
        snapshots, _ = self.gen.generate(positive=False, step_count=8)
        findings = [s.signals.findings_count for s in snapshots]
        assert findings == sorted(findings)
        assert findings[-1] > findings[0]

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValueError, match="step_count must be >= 3"):
            self.gen.generate(step_count=2)

    def test_fast_rise_burns_more_tokens(self) -> None:
        slow, _ = self.gen.generate(
            step_count=8,
            stuck_start=3,
            token_burn_pattern="slow_rise",
        )
        fast, _ = self.gen.generate(
            step_count=8,
            stuck_start=3,
            token_burn_pattern="fast_rise",
        )
        slow_final = slow[-1].signals.total_tokens
        fast_final = fast[-1].signals.total_tokens
        assert fast_final > slow_final
