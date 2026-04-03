"""RunawayCostGenerator — produces traces with controlled runaway cost failure modes.

A runaway cost trace shows token usage growing geometrically or in large steps
while findings stagnate. The burn rate per step exceeds the multiplier threshold
(3.0x baseline), indicating the agent is spending more and more without output.

Gate conditions maintained during runaway steps to prevent cross-detection:
  - source_productive=True: sources >= 10, findings >= 5, coverage >= 0.5
    (prevents stuck/loop false positives — see corpus_schema.md §3.6)
  - cv_coverage > 0.3 (above stuck_cv_threshold)
  - dm_coverage > 0.15 (above stuck_dm_threshold)
  - Token burn curves start at 80% of burn_rate to ensure the first runaway
    step exceeds the 3.0x baseline detection threshold (see _runaway_step).
"""

from __future__ import annotations

from typing import Any

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata


class RunawayCostGenerator(TraceGenerator):
    """Generate synthetic runaway cost traces with parameterized token burn."""

    def generate(
        self,
        *,
        trace_id: str = "runaway-syn-001",
        total_steps: int = 8,
        onset_step: int = 3,
        cost_growth: str = "quadratic",
        burn_rate: float = 4.0,
        base_tokens_per_step: int = 500,
        positive: bool = True,
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a runaway cost trace.

        Args:
            trace_id: Unique trace identifier.
            total_steps: Total trace length.
            onset_step: Step where runaway behavior begins.
            cost_growth: Growth pattern — "linear", "quadratic", or "step".
            burn_rate: Token burn multiplier over baseline at peak.
            base_tokens_per_step: Base token count per step.
            positive: If True, generate runaway cost trace.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if total_steps < 3:
            raise ValueError("total_steps must be >= 3 (min_evidence_steps)")

        snapshots: list[VitalsSnapshot] = []
        cumulative_tokens = 0

        for step in range(total_steps):
            if positive and step >= onset_step:
                signals, metrics, step_tokens = self._runaway_step(
                    step, onset_step, total_steps, cost_growth,
                    burn_rate, base_tokens_per_step, cumulative_tokens,
                )
            else:
                signals, metrics, step_tokens = self._healthy_step(
                    step, total_steps, base_tokens_per_step, cumulative_tokens,
                )

            cumulative_tokens += step_tokens

            snapshots.append(VitalsSnapshot(
                mission_id=f"bench-{trace_id}",
                run_id=trace_id,
                loop_index=step,
                signals=signals,
                metrics=metrics,
                health_state="healthy",
                timestamp=self._make_timestamp(step),
            ))

        labels = self._default_labels()
        labels["runaway_cost"] = positive

        params: dict[str, Any] = {
            "total_steps": total_steps,
            "onset_step": onset_step,
            "cost_growth": cost_growth,
            "burn_rate": burn_rate,
            "positive": positive,
        }

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="RunawayCostGenerator",
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
        cumulative: int,
    ) -> tuple[RawSignals, TemporalMetricsResult, int]:
        """Normal step — steady token usage with progress."""
        progress = (step + 1) / total_steps
        step_tokens = base_tokens
        total = cumulative + step_tokens

        signals = RawSignals(
            findings_count=step + 3,    # Higher baseline for source_productive
            sources_count=step + 8,     # >= 10 for source_productive
            objectives_covered=min(step + 1, 4),
            coverage_score=round(progress * 0.85, 3),  # Natural growth, no floor
            confidence_score=round(0.4 + progress * 0.3, 3),
            total_tokens=total,
            prompt_tokens=int(total * 0.6),
            completion_tokens=int(total * 0.4),
            error_count=0,
            convergence_delta=round(0.1 + progress * 0.05, 4),
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.35, 4) if step > 0 else 0.0,  # Above stuck_cv_threshold
            cv_findings_rate=round(0.2, 4) if step > 0 else 0.0,
            dm_coverage=round(0.6 + progress * 0.15, 4),
            dm_findings=round(0.55 + progress * 0.2, 4),
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.5 + progress * 0.15, 4),
        )

        return signals, metrics, step_tokens

    def _runaway_step(
        self,
        step: int,
        onset_step: int,
        total_steps: int,
        cost_growth: str,
        burn_rate: float,
        base_tokens: int,
        cumulative: int,
    ) -> tuple[RawSignals, TemporalMetricsResult, int]:
        """Runaway step — token burn accelerates, findings stagnate."""
        steps_runaway = step - onset_step + 1

        # Token growth pattern — curves start at 80% of burn_rate to ensure
        # the first runaway step produces tokens exceeding the burn_rate_anomaly
        # detection threshold (3.0x baseline).  Without this floor, gradual
        # curves never trigger per-step burn_rate_anomaly because the growing
        # token deltas raise the baseline average, keeping the ratio below 3x.
        n_runaway = max(1, total_steps - onset_step)
        frac = steps_runaway / n_runaway
        if cost_growth == "linear":
            multiplier = burn_rate * (0.8 + 0.2 * frac)
        elif cost_growth == "quadratic":
            multiplier = burn_rate * (0.8 + 0.2 * frac * frac)
        else:  # "step" — sudden jump
            multiplier = burn_rate

        step_tokens = int(base_tokens * multiplier)
        total = cumulative + step_tokens

        # Findings continue from healthy baseline (step+3) with slow growth.
        # Must stay above source_productive thresholds (findings>=5, sources>=10,
        # coverage>=0.5) to prevent token_usage_variance_flat stuck triggers.
        findings = onset_step + 3 + max(0, steps_runaway // 2)
        progress = (step + 1) / total_steps
        # Floor at 0.5 to keep source_productive=True (requires coverage>=0.5)
        coverage = max(0.5, min(0.85, progress * 0.85))

        signals = RawSignals(
            findings_count=findings,
            sources_count=findings + 5,  # >= 10 for source_productive
            objectives_covered=min(onset_step + 1, 4),
            coverage_score=round(coverage, 3),
            confidence_score=round(0.35, 3),
            total_tokens=total,
            prompt_tokens=int(total * 0.6),
            completion_tokens=int(total * 0.4),
            error_count=0,
            convergence_delta=round(0.01, 4),  # Minimal convergence
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.35, 4),  # Above stuck_cv_threshold (0.3)
            cv_findings_rate=round(0.15, 4),
            dm_coverage=round(0.25, 4),  # Above stuck_dm_threshold (0.15)
            dm_findings=round(0.2, 4),
            qpf_tokens=round(0.3 + steps_runaway * 0.05, 4),  # Growing unfairness
            cs_effort=round(0.2, 4),  # Low — effort not translating
        )

        return signals, metrics, step_tokens
