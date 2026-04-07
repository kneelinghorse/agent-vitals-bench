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

from typing import Any, Literal

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from generators.base import TraceGenerator, TraceMetadata

ConfabPattern = Literal[
    "classic",
    "delayed_sharp",
    "delayed_gradual",
    "delayed_oscillating",
]

CONFAB_DELAYED_PATTERNS: tuple[ConfabPattern, ...] = (
    "delayed_sharp",
    "delayed_gradual",
    "delayed_oscillating",
)
CONFAB_PATTERNS: tuple[ConfabPattern, ...] = ("classic", *CONFAB_DELAYED_PATTERNS)


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
        pattern: ConfabPattern = "classic",
        recovery_step: int | None = None,
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
            pattern: Confabulation trajectory shape. Delayed patterns add a
                healthy preamble before the ratio collapse.
            recovery_step: For delayed negative variants, the step where
                source support resumes and the trace recovers.
            positive: If True, generate confabulation trace.

        Returns:
            (snapshots, metadata) tuple with ground truth labels.
        """
        if total_steps < 3:
            raise ValueError("total_steps must be >= 3 (min_evidence_steps)")
        if pattern not in CONFAB_PATTERNS:
            raise ValueError(f"unsupported confab pattern: {pattern}")
        if onset_step < 1 or onset_step >= total_steps - 1:
            raise ValueError("onset_step must leave room for post-onset evidence")
        if pattern != "classic" and onset_step < 2:
            raise ValueError("delayed confab patterns require onset_step >= 2")
        if recovery_step is not None:
            if pattern == "classic":
                raise ValueError("recovery_step is only valid for delayed confab patterns")
            if recovery_step <= onset_step:
                raise ValueError("recovery_step must be after onset_step")
            if recovery_step >= total_steps:
                raise ValueError("recovery_step must be < total_steps")
        if not positive and pattern != "classic" and recovery_step is None:
            raise ValueError("delayed negative variants require a recovery_step")

        snapshots: list[VitalsSnapshot] = []

        for step in range(total_steps):
            if pattern == "classic" and positive and step >= onset_step:
                signals, metrics = self._confab_step(
                    step,
                    onset_step,
                    total_steps,
                    source_finding_ratio,
                    confidence_inflation,
                    base_tokens_per_step,
                )
            elif pattern != "classic" and step < onset_step:
                signals, metrics = self._productive_healthy_step(
                    step,
                    total_steps,
                    base_tokens_per_step,
                )
            elif (
                pattern != "classic"
                and step >= onset_step
                and (positive or recovery_step is None or step < recovery_step)
            ):
                signals, metrics = self._delayed_confab_step(
                    step=step,
                    onset_step=onset_step,
                    total_steps=total_steps,
                    target_ratio=source_finding_ratio,
                    confidence_inflation=confidence_inflation,
                    base_tokens=base_tokens_per_step,
                    pattern=pattern,
                    positive=positive,
                )
            elif pattern != "classic":
                assert recovery_step is not None
                signals, metrics = self._recovery_step(
                    step=step,
                    onset_step=onset_step,
                    recovery_step=recovery_step,
                    total_steps=total_steps,
                    base_tokens=base_tokens_per_step,
                    pattern=pattern,
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
        labels["confabulation"] = positive

        params: dict[str, Any] = {
            "total_steps": total_steps,
            "onset_step": onset_step,
            "source_finding_ratio": source_finding_ratio,
            "confidence_inflation": confidence_inflation,
            "pattern": pattern,
            "recovery_step": recovery_step,
            "positive": positive,
        }

        notes = ""
        if pattern != "classic":
            notes = (
                f"{pattern} delayed-onset confab pattern"
                if positive
                else f"{pattern} delayed-onset near-miss that recovers"
            )

        metadata = TraceMetadata(
            trace_id=trace_id,
            generator="ConfabGenerator",
            labels=labels,
            params=params,
            onset_step=onset_step if positive else None,
            notes=notes,
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
            verified_sources_count=sources,
            unverified_sources_count=0,
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

    def _productive_healthy_step(
        self,
        step: int,
        total_steps: int,
        base_tokens: int,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Healthy preamble with enough source evidence to avoid stuck cross-triggers."""
        progress = (step + 1) / total_steps
        findings = step + 3
        sources = max(10, findings + 5)

        signals = RawSignals(
            findings_count=findings,
            sources_count=sources,
            objectives_covered=min(step + 1, 5),
            coverage_score=round(min(1.0, 0.35 + progress * 0.55), 3),
            confidence_score=round(0.32 + progress * 0.25, 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.58),
            completion_tokens=int(base_tokens * (step + 1) * 0.42),
            error_count=0,
            query_count=step + 2,
            unique_domains=min(step + 4, 10),
            convergence_delta=round(0.12 + progress * 0.08, 4),
            verified_sources_count=sources,
            unverified_sources_count=0,
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.18, 4) if step > 0 else 0.0,
            cv_findings_rate=round(0.16, 4) if step > 0 else 0.0,
            dm_coverage=round(0.62 + progress * 0.14, 4),
            dm_findings=round(0.58 + progress * 0.16, 4),
            qpf_tokens=round(0.62, 4),
            cs_effort=round(0.54 + progress * 0.12, 4),
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

        # In confab phase: frozen verified sources, growing unverified
        verified = min(frozen_sources, sources)
        unverified = max(0, findings - verified)

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
            verified_sources_count=verified,
            unverified_sources_count=unverified,
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

    def _delayed_confab_step(
        self,
        *,
        step: int,
        onset_step: int,
        total_steps: int,
        target_ratio: float,
        confidence_inflation: float,
        base_tokens: int,
        pattern: ConfabPattern,
        positive: bool,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Delayed confab phase with explicit sharp/gradual/oscillating collapse patterns."""
        progress = (step + 1) / total_steps
        steps_confab = step - onset_step
        growth = {
            "delayed_sharp": 12,
            "delayed_gradual": 8,
            "delayed_oscillating": 9,
        }[pattern]
        findings = onset_step + 3 + (steps_confab + 1) * growth

        effective_target = target_ratio if positive else max(0.38, target_ratio + 0.18)
        ratio = self._pattern_ratio(
            pattern=pattern,
            step_index=steps_confab,
            target_ratio=effective_target,
        )
        sources = max(10, int(round(findings * ratio)))

        verified_ratio = 0.28 if positive else 0.62
        min_verified = 3 if positive else 6
        verified = min(sources, max(min_verified, int(round(sources * verified_ratio))))
        unverified = max(0, sources - verified)

        base_confidence = 0.35 + progress * 0.25
        confidence = (
            min(1.0, base_confidence + confidence_inflation)
            if positive
            else min(0.78, base_confidence + max(0.0, confidence_inflation - 0.05))
        )

        domains = min(onset_step + 4, 7) if positive else min(onset_step + 5, 8)
        convergence = 0.03 if positive else 0.06

        signals = RawSignals(
            findings_count=findings,
            sources_count=sources,
            objectives_covered=min(step + 1, 5),
            coverage_score=round(max(0.55, min(1.0, 0.48 + progress * 0.4)), 3),
            confidence_score=round(confidence, 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.58),
            completion_tokens=int(base_tokens * (step + 1) * 0.42),
            error_count=0,
            query_count=onset_step + 2,
            unique_domains=domains,
            convergence_delta=round(convergence, 4),
            verified_sources_count=verified,
            unverified_sources_count=unverified,
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.14 if positive else 0.12, 4),
            cv_findings_rate=round(0.32 + steps_confab * 0.04, 4),
            dm_coverage=round(0.42 if positive else 0.5, 4),
            dm_findings=round(min(1.0, 0.74 + steps_confab * 0.03), 4),
            qpf_tokens=round(0.64, 4),
            cs_effort=round(0.42 if positive else 0.48, 4),
        )

        return signals, metrics

    def _recovery_step(
        self,
        *,
        step: int,
        onset_step: int,
        recovery_step: int,
        total_steps: int,
        base_tokens: int,
        pattern: ConfabPattern,
    ) -> tuple[RawSignals, TemporalMetricsResult]:
        """Recovery phase for delayed-onset hard negatives."""
        progress = (step + 1) / total_steps
        recovery_index = step - recovery_step
        growth = {
            "delayed_sharp": 12,
            "delayed_gradual": 8,
            "delayed_oscillating": 9,
        }[pattern]
        failure_steps = recovery_step - onset_step
        findings_at_recovery = onset_step + 3 + max(1, failure_steps) * growth
        findings = findings_at_recovery + (recovery_index + 1) * 4
        recovery_ratio = min(1.15, 0.55 + recovery_index * 0.18)
        sources = max(12, int(round(findings * recovery_ratio)))
        verified = min(sources, max(8, int(round(sources * 0.82))))
        unverified = max(0, sources - verified)

        signals = RawSignals(
            findings_count=findings,
            sources_count=sources,
            objectives_covered=min(step + 1, 5),
            coverage_score=round(min(1.0, 0.62 + progress * 0.25), 3),
            confidence_score=round(max(0.45, 0.72 - recovery_index * 0.08), 3),
            total_tokens=base_tokens * (step + 1),
            prompt_tokens=int(base_tokens * (step + 1) * 0.58),
            completion_tokens=int(base_tokens * (step + 1) * 0.42),
            error_count=0,
            query_count=step + 2,
            unique_domains=min(8 + recovery_index, 12),
            convergence_delta=round(0.11 + recovery_index * 0.015, 4),
            verified_sources_count=verified,
            unverified_sources_count=unverified,
        )

        metrics = TemporalMetricsResult(
            cv_coverage=round(0.11, 4),
            cv_findings_rate=round(0.2, 4),
            dm_coverage=round(0.61 + progress * 0.1, 4),
            dm_findings=round(0.62 + recovery_index * 0.02, 4),
            qpf_tokens=round(0.63, 4),
            cs_effort=round(0.56 + recovery_index * 0.03, 4),
        )

        return signals, metrics

    @staticmethod
    def _pattern_ratio(
        *,
        pattern: ConfabPattern,
        step_index: int,
        target_ratio: float,
    ) -> float:
        """Return the source-to-finding ratio for a delayed confab sub-pattern."""
        if pattern == "delayed_sharp":
            schedule = [
                max(0.9, target_ratio + 0.65),
                max(0.48, target_ratio + 0.26),
                max(0.26, target_ratio + 0.06),
                max(target_ratio + 0.02, target_ratio),
                target_ratio,
            ]
        elif pattern == "delayed_gradual":
            schedule = [
                1.05,
                max(0.82, target_ratio + 0.48),
                max(0.6, target_ratio + 0.3),
                max(0.42, target_ratio + 0.14),
                max(0.3, target_ratio + 0.04),
                target_ratio,
            ]
        else:
            schedule = [
                max(0.95, target_ratio + 0.58),
                max(0.58, target_ratio + 0.2),
                max(0.8, target_ratio + 0.44),
                max(0.46, target_ratio + 0.12),
                max(0.26, target_ratio + 0.04),
                target_ratio,
            ]

        return float(schedule[min(step_index, len(schedule) - 1)])
