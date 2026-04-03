"""StuckGenerator — produces traces with controlled stuck failure modes.

A stuck trace shows the agent burning tokens while making zero progress.
The key signals are: DM (directional momentum) near zero, CV (coefficient
of variation) near zero, and total_tokens increasing. Unlike loops, stuck
agents don't even repeat — they simply stall.

Gate conditions for stuck detection (from loop.py):
  - dm_coverage <= stuck_dm_threshold (0.15) for >= 3 consecutive steps
  - cv_coverage <= stuck_cv_threshold (0.3)
  - source_productive must be False (sources < 10, findings < 5, or coverage < 0.5)
  - coverage_score < 0.95 (high coverage = convergence, not stuck)
  - Workflow must be research-only or workflow_stuck_enabled=all
"""

from __future__ import annotations

from typing import Any

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata


class StuckGenerator(TraceGenerator):
    """Generate synthetic stuck traces with parameterized stall behavior."""

    def generate(
        self,
        *,
        trace_id: str = "stuck-syn-001",
        step_count: int = 8,
        stuck_start: int = 3,
        dm_value: float = 0.05,
        cv_value: float = 0.1,
        token_burn_pattern: str = "slow_rise",
        base_tokens_per_step: int = 500,
        positive: bool = True,
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a stuck trace.

        Args:
            trace_id: Unique trace identifier.
            step_count: Total trace length.
            stuck_start: Step where stall begins (0-based).
            dm_value: DM value during stuck phase (should be < stuck_dm_threshold=0.15).
            cv_value: CV value during stuck phase (should be < stuck_cv_threshold=0.3).
            token_burn_pattern: "flat", "slow_rise", or "fast_rise".
            base_tokens_per_step: Base token count per step.
            positive: If True, generate stuck trace. If False, healthy trace.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if step_count < 3:
            raise ValueError("step_count must be >= 3 (min_evidence_steps)")

        snapshots: list[VitalsSnapshot] = []

        for step in range(step_count):
            if positive and step >= stuck_start:
                signals, metrics = self._stuck_step(
                    step, stuck_start, dm_value, cv_value,
                    token_burn_pattern, base_tokens_per_step, step_count,
                )
            else:
                signals, metrics = self._progress_step(
                    step, step_count, base_tokens_per_step,
                )

            # Stuck phase: low output_similarity (agent flails, not repeating)
            sim = 0.15 if (positive and step >= stuck_start) else None

            snapshots.append(VitalsSnapshot(
                mission_id=f"bench-{trace_id}",
                run_id=trace_id,
                loop_index=step,
                signals=signals,
                metrics=metrics,
                health_state="healthy",
                timestamp=self._make_timestamp(step),
                output_similarity=sim,
            ))

        labels = self._default_labels()
        labels["stuck"] = positive

        params: dict[str, Any] = {
            "step_count": step_count,
            "stuck_start": stuck_start,
            "dm_value": dm_value,
            "cv_value": cv_value,
            "token_burn_pattern": token_burn_pattern,
            "positive": positive,
        }

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="StuckGenerator",
            labels=labels,
            params=params,
            onset_step=stuck_start if positive else None,
        )

        return snapshots, metadata

    def _progress_step(
        self,
        step: int,
        total_steps: int,
        base_tokens: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Normal progress step — findings and coverage growing."""
        progress = (step + 1) / total_steps
        findings = step + 1
        coverage = min(1.0, progress * 0.85)
        tokens = base_tokens * (step + 1)

        signals = RawSignals(
            findings_count=findings,
            sources_count=findings + 3,
            objectives_covered=min(step + 1, 4),
            coverage_score=round(coverage, 3),
            confidence_score=round(0.4 + progress * 0.4, 3),
            total_tokens=tokens,
            prompt_tokens=int(tokens * 0.55),
            completion_tokens=int(tokens * 0.45),
            error_count=0,
            convergence_delta=round(0.15 + progress * 0.05, 4),
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.35 - progress * 0.1, 4) if step > 0 else 0.0,
            cv_findings_rate=round(0.25, 4) if step > 0 else 0.0,
            dm_coverage=round(0.6 + progress * 0.15, 4),
            dm_findings=round(0.55 + progress * 0.2, 4),
            qpf_tokens=round(0.55, 4),
            cs_effort=round(0.5 + progress * 0.15, 4),
        )

        return signals, metrics

    def _stuck_step(
        self,
        step: int,
        stuck_start: int,
        dm_value: float,
        cv_value: float,
        token_burn_pattern: str,
        base_tokens: int,
        total_steps: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Stuck step — near-zero progress, tokens burning."""
        # Findings grow very slowly — enough to prevent findings_plateau
        # detection (which needs consecutive zero deltas) but slow enough
        # that DM/CV remain the dominant stuck signals.
        steps_stuck = step - stuck_start
        frozen_findings = stuck_start + max(0, steps_stuck // 3)
        frozen_coverage = round(
            stuck_start / total_steps * 0.85 + steps_stuck * 0.005, 3
        )

        # Token burn depends on pattern
        if token_burn_pattern == "flat":
            tokens = base_tokens * (stuck_start + 1) + base_tokens * (steps_stuck + 1)
        elif token_burn_pattern == "slow_rise":
            tokens = base_tokens * (step + 1) + int(steps_stuck * base_tokens * 0.3)
        else:  # fast_rise
            tokens = base_tokens * (step + 1) + int(steps_stuck ** 2 * base_tokens * 0.5)

        signals = RawSignals(
            findings_count=frozen_findings,
            sources_count=frozen_findings + 3,
            objectives_covered=min(stuck_start, 4),
            coverage_score=frozen_coverage,
            confidence_score=round(0.4, 3),
            total_tokens=tokens,
            prompt_tokens=int(tokens * 0.55),
            completion_tokens=int(tokens * 0.45),
            error_count=0,
            convergence_delta=0.0,  # Zero convergence — stuck
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(cv_value, 4),
            cv_findings_rate=round(cv_value * 0.5, 4),
            dm_coverage=round(dm_value, 4),
            dm_findings=round(dm_value * 0.8, 4),
            qpf_tokens=round(0.55, 4),
            cs_effort=round(0.2, 4),  # Low — effort not producing results
        )

        return signals, metrics
