"""ThrashGenerator — produces traces with controlled thrash failure modes.

A thrash trace shows the agent oscillating between approaches: error spikes,
approach switches (objectives_covered going up then down), and high refinement
counts. The agent is working but unable to settle on a direction.

Gate conditions for thrash detection (from stop_rule.py):
  - error_count >= thrash_error_threshold (1) AND
  - refinement_count increases AND
  - objectives_covered oscillates (up then down)
  - Trace length >= min_steps_for_thrash (3)
  - Errors suppress loop detection, so thrash and loop are mutually exclusive
"""

from __future__ import annotations

from typing import Any

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata


class ThrashGenerator(TraceGenerator):
    """Generate synthetic thrash traces with error spikes and approach oscillation."""

    def generate(
        self,
        *,
        trace_id: str = "thrash-syn-001",
        total_steps: int = 8,
        onset_step: int = 2,
        error_spikes: int = 3,
        refinement_growth: int = 2,
        base_tokens_per_step: int = 500,
        positive: bool = True,
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a thrash trace.

        Args:
            trace_id: Unique trace identifier.
            total_steps: Total trace length.
            onset_step: Step where thrash behavior begins.
            error_spikes: Number of error spikes during thrash phase.
            refinement_growth: Refinement count increase per thrash step.
            base_tokens_per_step: Base token count per step.
            positive: If True, generate thrash trace.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if total_steps < 3:
            raise ValueError("total_steps must be >= 3 (min_evidence_steps)")

        snapshots: list[VitalsSnapshot] = []

        for step in range(total_steps):
            if positive and step >= onset_step:
                signals, metrics = self._thrash_step(
                    step,
                    onset_step,
                    total_steps,
                    error_spikes,
                    refinement_growth,
                    base_tokens_per_step,
                )
            else:
                signals, metrics = self._healthy_step(
                    step,
                    total_steps,
                    base_tokens_per_step,
                )

            snapshots.append(
                VitalsSnapshot(
                    mission_id=f"bench-{trace_id}",
                    run_id=trace_id,
                    loop_index=step,
                    signals=signals,
                    metrics=metrics,
                    health_state="healthy",
                    timestamp=self._make_timestamp(step),
                )
            )

        labels = self._default_labels()
        labels["thrash"] = positive

        params: dict[str, Any] = {
            "total_steps": total_steps,
            "onset_step": onset_step,
            "error_spikes": error_spikes,
            "refinement_growth": refinement_growth,
            "positive": positive,
        }

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="ThrashGenerator",
            labels=labels,
            params=params,
            onset_step=onset_step if positive else None,
        )

        return snapshots, metadata

    def _healthy_step(
        self,
        step: int,
        total_steps: int,
        base_tokens: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Normal step — steady progress, no errors."""
        progress = (step + 1) / total_steps

        signals = RawSignals(
            findings_count=step + 1,
            sources_count=step + 3,
            objectives_covered=min(step + 1, 4),
            coverage_score=round(progress * 0.8, 3),
            confidence_score=round(0.4 + progress * 0.3, 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.6),
            completion_tokens=int(base_tokens * (step + 1) * 0.4),
            error_count=0,
            refinement_count=0,
            convergence_delta=round(0.12 + progress * 0.03, 4),
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.25, 4) if step > 0 else 0.0,
            cv_findings_rate=round(0.2, 4) if step > 0 else 0.0,
            dm_coverage=round(0.6 + progress * 0.15, 4),
            dm_findings=round(0.55 + progress * 0.2, 4),
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.5 + progress * 0.15, 4),
        )

        return signals, metrics

    def _thrash_step(
        self,
        step: int,
        onset_step: int,
        total_steps: int,
        error_spikes: int,
        refinement_growth: int,
        base_tokens: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Thrash step — errors spike, objectives oscillate, refinements climb."""
        steps_thrash = step - onset_step

        # Error accumulation with spikes
        errors_per_step = max(1, error_spikes // max(1, total_steps - onset_step))
        error_count = (steps_thrash + 1) * errors_per_step

        # Refinement count climbing
        refinement_count = (steps_thrash + 1) * refinement_growth

        # Objectives oscillate (approach switching)
        # Pattern: goes up then down then up...
        base_objectives = min(onset_step + 1, 3)
        oscillation = [2, -1, 1, -2, 3, -1, 2, -1]
        obj_delta = oscillation[steps_thrash % len(oscillation)]
        objectives = max(1, base_objectives + obj_delta)

        # Findings barely advance (switching approaches wastes effort)
        findings = onset_step + 1 + max(0, steps_thrash // 3)

        # Coverage oscillates too
        base_coverage = onset_step / total_steps * 0.8
        coverage_noise = 0.05 * (1 if steps_thrash % 2 == 0 else -1)

        signals = RawSignals(
            findings_count=findings,
            sources_count=findings + 2,
            objectives_covered=objectives,
            coverage_score=round(min(1.0, max(0.0, base_coverage + coverage_noise)), 3),
            confidence_score=round(0.3, 3),  # Low confidence — unsettled
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.6),
            completion_tokens=int(base_tokens * (step + 1) * 0.4),
            error_count=error_count,
            refinement_count=refinement_count,
            convergence_delta=round(0.01, 4),  # Near-zero convergence
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.5 + steps_thrash * 0.05, 4),  # High variance
            cv_findings_rate=round(0.4, 4),
            dm_coverage=round(0.2, 4),  # Above stuck_dm_threshold (0.15)
            dm_findings=round(0.2, 4),
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.25, 4),  # Low symmetry — erratic effort
        )

        return signals, metrics
