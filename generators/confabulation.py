"""ConfabGenerator — produces traces with controlled confabulation failure modes.

A confabulation trace shows findings growing while sources stagnate or decline,
creating a source-finding ratio that drops below the floor threshold (0.3).
The agent appears productive (findings up, confidence high) but the findings
are unsupported by actual sources — the hallmark of fabricated output.

Gate conditions for confabulation detection (from loop.py):
  - source_finding_ratio < source_finding_ratio_floor (0.3) AND declining for
    >= source_finding_ratio_declining_steps (3) consecutive steps
  - OR confidence_score >= 0.85 with sources_count = 0
  - Boosted by CUSUM findings_count_delta alarm and SPC ratio threshold
  - Sources stagnation: sources flat while findings grow (SOURCES_STAGNATION_WINDOW=2)
  - Unique domains stagnation: all <= 1 in trailing window (UNIQUE_DOMAINS_STAGNATION_WINDOW=3)
"""

from __future__ import annotations

from typing import Any

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata


class ConfabGenerator(TraceGenerator):
    """Generate synthetic confabulation traces with parameterized ratio collapse."""

    def generate(
        self,
        *,
        trace_id: str = "confab-syn-001",
        total_steps: int = 10,
        onset_step: int = 3,
        source_finding_ratio: float = 0.2,
        confidence_inflation: float = 0.3,
        base_tokens_per_step: int = 500,
        positive: bool = True,
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a confabulation trace.

        Args:
            trace_id: Unique trace identifier.
            total_steps: Total trace length.
            onset_step: Step where confabulation behavior begins.
            source_finding_ratio: Target source/finding ratio in confab phase
                (should be < source_finding_ratio_floor=0.3 for positives).
            confidence_inflation: Amount to inflate confidence_score during
                confab phase (high confidence with no sources).
            base_tokens_per_step: Base token count per step.
            positive: If True, generate confabulation trace.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if total_steps < 3:
            raise ValueError("total_steps must be >= 3 (min_evidence_steps)")

        snapshots: list[VitalsSnapshot] = []

        for step in range(total_steps):
            if positive and step >= onset_step:
                signals, metrics = self._confab_step(
                    step, onset_step, total_steps, source_finding_ratio,
                    confidence_inflation, base_tokens_per_step,
                )
            else:
                signals, metrics = self._healthy_step(
                    step, total_steps, base_tokens_per_step,
                )

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
        labels["confabulation"] = positive

        params: dict[str, Any] = {
            "total_steps": total_steps,
            "onset_step": onset_step,
            "source_finding_ratio": source_finding_ratio,
            "confidence_inflation": confidence_inflation,
            "positive": positive,
        }

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="ConfabGenerator",
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
        """Normal step — sources and findings grow together."""
        progress = (step + 1) / total_steps
        findings = step + 2
        sources = findings + 3  # Healthy ratio: sources > findings

        signals = RawSignals(
            findings_count=findings,
            sources_count=sources,
            objectives_covered=min(step + 1, 4),
            coverage_score=round(progress * 0.8, 3),
            confidence_score=round(0.3 + progress * 0.3, 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.6),
            completion_tokens=int(base_tokens * (step + 1) * 0.4),
            error_count=0,
            query_count=step + 1,
            unique_domains=min(step + 2, 6),
            convergence_delta=round(0.1 + progress * 0.05, 4),
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

    def _confab_step(
        self,
        step: int,
        onset_step: int,
        total_steps: int,
        target_ratio: float,
        confidence_inflation: float,
        base_tokens: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Confabulation step — findings grow, sources stagnate."""
        progress = (step + 1) / total_steps
        steps_confab = step - onset_step

        # Findings keep growing (agent is "productive")
        findings = onset_step + 2 + (steps_confab + 1) * 3

        # Sources freeze at onset level (fabricating without real sources)
        frozen_sources = onset_step + 5
        # Compute sources to hit target ratio approximately
        target_sources = max(1, int(findings * target_ratio))
        sources = min(frozen_sources, target_sources)

        # Confidence inflated (agent is "confident" despite no sources)
        base_confidence = 0.3 + progress * 0.3
        confidence = min(1.0, base_confidence + confidence_inflation)

        # Domains stagnate (not exploring new territory)
        domains = min(onset_step + 2, 4)

        signals = RawSignals(
            findings_count=findings,
            sources_count=sources,
            objectives_covered=min(step + 1, 5),
            coverage_score=round(min(1.0, progress * 0.85), 3),
            confidence_score=round(confidence, 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.6),
            completion_tokens=int(base_tokens * (step + 1) * 0.4),
            error_count=0,
            query_count=onset_step + 1,  # Queries also stagnate
            unique_domains=domains,
            convergence_delta=round(0.02, 4),  # Low convergence
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.15, 4),
            cv_findings_rate=round(0.35 + steps_confab * 0.05, 4),  # Growing variance
            dm_coverage=round(0.4, 4),
            dm_findings=round(0.7 + steps_confab * 0.03, 4),  # Findings momentum high
            qpf_tokens=round(0.6, 4),
            cs_effort=round(0.4, 4),
        )

        return signals, metrics
