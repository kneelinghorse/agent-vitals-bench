"""LoopGenerator — produces traces with controlled loop failure modes.

A loop trace has a normal progress phase followed by a plateau phase where
findings_count stops increasing but tokens continue to burn. The generator
parameters control exactly when and how the loop manifests, providing
perfect ground truth for evaluator calibration.

Gate conditions for loop detection (from loop.py):
  - findings_count plateau for >= loop_consecutive_pct * trace_length steps
  - OR coverage_score flat (< 0.01 delta) for >= loop_consecutive_pct steps
  - OR output_similarity >= loop_similarity_threshold (0.8)
  - source_productive must be False for plateau/flat triggers
  - Errors suppress loop detection (errors present = thrash, not loop)
  - Pattern-specific metrics: cv_coverage 0.32 (exact), 0.35 (semantic/partial);
    dm_coverage 0.12 (below stuck threshold 0.15 to simulate near-stuck overlap)
"""

from __future__ import annotations

from typing import Any

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata


class LoopGenerator(TraceGenerator):
    """Generate synthetic loop traces with parameterized onset and pattern."""

    def generate(
        self,
        *,
        trace_id: str = "loop-syn-001",
        loop_start: int = 3,
        loop_length: int = 4,
        total_steps: int = 10,
        pattern: str = "exact",
        base_tokens_per_step: int = 500,
        positive: bool = True,
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a loop trace.

        Args:
            trace_id: Unique trace identifier.
            loop_start: Step index (0-based) where loop begins.
            loop_length: Number of looping steps after onset.
            total_steps: Total trace length.
            pattern: Loop pattern — "exact" (identical signals), "semantic"
                (slight variation), or "partial" (some progress leaks through).
            base_tokens_per_step: Base token count per step.
            positive: If True, generate a positive (looping) trace.
                If False, generate a healthy trace with steady progress.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if total_steps < 3:
            raise ValueError("total_steps must be >= 3 (min_evidence_steps)")
        if positive and loop_start + loop_length > total_steps:
            raise ValueError("loop_start + loop_length exceeds total_steps")

        snapshots: list[VitalsSnapshot] = []

        for step in range(total_steps):
            if positive and step >= loop_start:
                signals, metrics = self._loop_step(
                    step,
                    loop_start,
                    pattern,
                    base_tokens_per_step,
                    total_steps,
                )
            else:
                signals, metrics = self._progress_step(
                    step,
                    total_steps,
                    base_tokens_per_step,
                    positive,
                )

            # Loop phase: high output_similarity (agent repeats content)
            sim = 0.92 if (positive and step >= loop_start) else None

            snapshots.append(
                VitalsSnapshot(
                    mission_id=f"bench-{trace_id}",
                    run_id=trace_id,
                    loop_index=step,
                    signals=signals,
                    metrics=metrics,
                    health_state="healthy",
                    timestamp=self._make_timestamp(step),
                    output_similarity=sim,
                )
            )

        labels = self._default_labels()
        labels["loop"] = positive

        params: dict[str, Any] = {
            "loop_start": loop_start,
            "loop_length": loop_length,
            "total_steps": total_steps,
            "pattern": pattern,
            "positive": positive,
        }

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="LoopGenerator",
            labels=labels,
            params=params,
            onset_step=loop_start if positive else None,
        )

        return snapshots, metadata

    def _progress_step(
        self,
        step: int,
        total_steps: int,
        base_tokens: int,
        steady: bool,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Generate a normal progress step with increasing findings/coverage."""
        progress = (step + 1) / total_steps
        findings = step + 1 if steady else max(1, int(progress * total_steps * 0.8))
        coverage = min(1.0, progress * 0.9)
        tokens = base_tokens * (step + 1)

        signals = RawSignals(
            findings_count=findings,
            sources_count=findings + 2,
            objectives_covered=min(step + 1, 5),
            coverage_score=round(coverage, 3),
            confidence_score=round(min(0.9, 0.3 + progress * 0.6), 3),
            total_tokens=tokens,
            prompt_tokens=int(tokens * 0.6),
            completion_tokens=int(tokens * 0.4),
            error_count=0,
            convergence_delta=round(0.1 + progress * 0.05, 4),
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.3 - progress * 0.1, 4) if step > 0 else 0.0,
            cv_findings_rate=round(0.2 - progress * 0.05, 4) if step > 0 else 0.0,
            dm_coverage=round(0.6 + progress * 0.2, 4),
            dm_findings=round(0.5 + progress * 0.3, 4),
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.5 + progress * 0.2, 4),
        )

        return signals, metrics

    def _loop_step(
        self,
        step: int,
        loop_start: int,
        pattern: str,
        base_tokens: int,
        total_steps: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Generate a looping step with plateau characteristics."""
        # Findings freeze at the value from loop_start
        frozen_findings = loop_start
        frozen_coverage = round(loop_start / total_steps * 0.9, 3)

        # Tokens keep burning (agent is still working, just not progressing)
        tokens = base_tokens * (step + 1)

        # Pattern variations
        noise = 0
        if pattern == "semantic":
            # Slight variation but no real progress
            noise = step % 2  # 0 or 1 variation in findings
        elif pattern == "partial":
            # Occasional progress leak (makes detection harder)
            noise = 1 if step % 3 == 0 else 0

        signals = RawSignals(
            findings_count=frozen_findings + noise,
            sources_count=frozen_findings + noise + 2,
            objectives_covered=min(loop_start, 5),
            coverage_score=min(1.0, frozen_coverage + noise * 0.01),
            confidence_score=round(0.5, 3),
            total_tokens=tokens,
            prompt_tokens=int(tokens * 0.6),
            completion_tokens=int(tokens * 0.4),
            error_count=0,
            convergence_delta=0.0,
        )

        metrics = TemporalMetricsResult(
            cv_coverage=0.32 if pattern == "exact" else 0.35,  # Above stuck_cv_threshold (0.3)
            cv_findings_rate=0.05 if pattern == "exact" else 0.10,
            dm_coverage=0.12,  # Below stuck_dm_threshold but cv above prevents stuck
            dm_findings=0.05,
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.3, 4),  # Low — effort not translating to progress
        )

        return signals, metrics
