"""Regression tests for stuck detection on fast_rise token-burn traces.

The corpus v1 stuck family includes 33 synthetic traces with
token_burn_pattern="fast_rise" — these exercise the rapid-token-burn
code path where total_tokens grows quadratically during the stuck
phase (steps_stuck**2 * base_tokens * 0.5).  During Sprint 17 parity
work, this family was the source of a replay-drift scare; these tests
lock the detection contract so any future regression is caught.
"""

from __future__ import annotations

import pytest
from agent_vitals import VitalsConfig

from evaluator.runner import load_manifest, load_trace, replay_trace, resolve_workflow_type
from generators.stuck import StuckGenerator


def _fast_rise_entries() -> list[dict]:
    """Return manifest entries with token_burn_pattern=fast_rise."""
    manifest = load_manifest("v1")
    return [
        e
        for e in manifest
        if e.get("metadata", {}).get("params", {}).get("token_burn_pattern") == "fast_rise"
        and float(e.get("metadata", {}).get("confidence", 0.0) or 0.0) >= 0.8
    ]


class TestStuckFastRiseCorpus:
    """Replay actual fast_rise corpus traces and assert correct stuck detection."""

    @pytest.fixture(autouse=True)
    def _config(self) -> None:
        self.config = VitalsConfig.from_yaml(allow_env_override=False)

    def test_fast_rise_family_exists(self) -> None:
        """Sanity: the fast_rise trace family is present in corpus v1."""
        entries = _fast_rise_entries()
        assert len(entries) >= 24, f"Expected >= 24 fast_rise traces, found {len(entries)}"

    def test_all_fast_rise_positives_detected(self) -> None:
        """Every fast_rise positive trace must trigger stuck detection."""
        entries = _fast_rise_entries()
        positives = [e for e in entries if e["labels"]["stuck"] is True]
        assert len(positives) >= 24

        missed: list[str] = []
        for entry in positives:
            snapshots = load_trace("v1", entry["path"])
            mission_id = snapshots[0].mission_id if snapshots else None
            wf_type = resolve_workflow_type(entry["trace_id"], mission_id)
            predictions = replay_trace(
                snapshots, config=self.config, workflow_type=wf_type, runtime_mode="default"
            )
            if not predictions["stuck"]:
                missed.append(entry["trace_id"])

        assert not missed, f"stuck detector missed {len(missed)} fast_rise positives: {missed}"

    def test_no_cross_detector_leakage(self) -> None:
        """fast_rise stuck traces should not trigger loop, confab, or thrash."""
        entries = _fast_rise_entries()
        positives = [e for e in entries if e["labels"]["stuck"] is True]
        leaks: list[str] = []

        for entry in positives:
            snapshots = load_trace("v1", entry["path"])
            mission_id = snapshots[0].mission_id if snapshots else None
            wf_type = resolve_workflow_type(entry["trace_id"], mission_id)
            predictions = replay_trace(
                snapshots, config=self.config, workflow_type=wf_type, runtime_mode="default"
            )
            # Check for spurious detections on detectors that should NOT fire
            for detector in ("loop", "confabulation", "thrash"):
                if predictions[detector] and not entry["labels"].get(detector, False):
                    leaks.append(f"{entry['trace_id']}:{detector}")

        assert not leaks, f"Cross-detector leakage on fast_rise stuck traces: {leaks}"


class TestStuckFastRiseSynthetic:
    """Generate fresh fast_rise traces and verify detection contract."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.gen = StuckGenerator()
        self.config = VitalsConfig.from_yaml(allow_env_override=False)

    def test_generated_positive_detected(self) -> None:
        """A freshly generated fast_rise stuck trace must be detected."""
        snapshots, meta = self.gen.generate(
            step_count=8,
            stuck_start=3,
            token_burn_pattern="fast_rise",
            positive=True,
        )
        predictions = replay_trace(
            snapshots, config=self.config, workflow_type="unknown", runtime_mode="default"
        )
        assert predictions["stuck"], "Stuck detector failed on generated fast_rise positive"

    def test_generated_negative_not_detected(self) -> None:
        """A fast_rise trace with healthy metrics should NOT trigger stuck."""
        snapshots, meta = self.gen.generate(
            step_count=8,
            stuck_start=3,
            token_burn_pattern="fast_rise",
            positive=False,
        )
        predictions = replay_trace(
            snapshots, config=self.config, workflow_type="unknown", runtime_mode="default"
        )
        assert not predictions["stuck"], "Stuck detector false-fired on fast_rise negative"

    @pytest.mark.parametrize("step_count", [5, 8, 12])
    def test_detection_stable_across_trace_lengths(self, step_count: int) -> None:
        """fast_rise detection should work regardless of trace length."""
        snapshots, _ = self.gen.generate(
            step_count=step_count,
            stuck_start=2,
            token_burn_pattern="fast_rise",
            positive=True,
        )
        predictions = replay_trace(
            snapshots, config=self.config, workflow_type="unknown", runtime_mode="default"
        )
        assert predictions["stuck"], (
            f"Stuck detector missed fast_rise positive at step_count={step_count}"
        )
