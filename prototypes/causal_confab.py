"""Causal confabulation prototype based on rolling source-link degradation.

The prototype models two parallel causal links with rolling, token-controlled
partial correlation scores. The detector fires when any of these conditions hold:

1. A healthy early findings->sources link breaks later in the trace, or
2. The trace starts with a small source bootstrap and never establishes a
   meaningful source-growth link.
3. Verified sources decouple from total source growth — real LLM confabulation
   where each hallucinated finding comes with a hallucinated source, but DOI
   verification reveals that verified sources stagnate while total sources grow.

Paths 1-2 catch synthetic confabulation (findings/sources decoupling).
Path 3 catches real LLM confabulation (verified/total source decoupling).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

from agent_vitals.schema import VitalsSnapshot

from evaluator.metrics import DetectorMetrics


@dataclass(frozen=True, slots=True)
class CausalConfabConfig:
    """Tunable parameters for the causal confabulation prototype."""

    window_size: int = 4
    baseline_floor: float = 0.4
    weak_link_threshold: float = 0.35
    structural_drop_threshold: float = 0.2
    ratio_gate: float = 0.4
    low_link_threshold: float = 0.2
    source_bootstrap_cap: int = 10
    low_link_ratio_gate: float = 0.3
    # Path 3: verified source decoupling (for elicited traces with DOI data)
    # verified_link_floor: max verified_link_strength must be below this
    # (positives max out at ~0.45, so 0.5 catches all while staying conservative)
    verified_link_floor: float = 0.5
    verified_weak_threshold: float = 0.25
    verified_drop_threshold: float = 0.15
    # verified_ratio_gate: final verified/total ratio must be below this
    # (matches elicitation positive label threshold of 0.3)
    verified_ratio_gate: float = 0.3
    verified_min_sources: int = 10
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3 snapshots")


@dataclass(frozen=True, slots=True)
class WindowScore:
    """Rolling causal-link score for a single window of snapshots."""

    start_step: int
    end_step: int
    link_strength: float
    response_ratio: float
    partial_correlation: float | None
    source_finding_ratio: float
    verified_link_strength: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "link_strength": round(self.link_strength, 4),
            "response_ratio": round(self.response_ratio, 4),
            "partial_correlation": (
                None if self.partial_correlation is None else round(self.partial_correlation, 4)
            ),
            "source_finding_ratio": round(self.source_finding_ratio, 4),
        }
        if self.verified_link_strength is not None:
            result["verified_link_strength"] = round(self.verified_link_strength, 4)
        return result


@dataclass(frozen=True, slots=True)
class CausalConfabResult:
    """Detection output for a single trace."""

    detected: bool
    confidence: float
    trigger: str | None
    baseline_strength: float
    weakest_strength: float
    structural_drop: float
    final_ratio: float
    initial_sources: int
    onset_step: int | None
    window_scores: tuple[WindowScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "confidence": round(self.confidence, 4),
            "trigger": self.trigger,
            "baseline_strength": round(self.baseline_strength, 4),
            "weakest_strength": round(self.weakest_strength, 4),
            "structural_drop": round(self.structural_drop, 4),
            "final_ratio": round(self.final_ratio, 4),
            "initial_sources": self.initial_sources,
            "onset_step": self.onset_step,
            "window_scores": [score.as_dict() for score in self.window_scores],
        }


@dataclass(frozen=True, slots=True)
class CausalConfabEvaluation:
    """Aggregate corpus-level evaluation for the prototype."""

    corpus_version: str
    trace_count: int
    metric: DetectorMetrics
    config: CausalConfabConfig
    min_confidence: float
    tier: str | None
    confab_only: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_version": self.corpus_version,
            "trace_count": self.trace_count,
            "metric": self.metric.as_dict(),
            "config": asdict(self.config),
            "min_confidence": self.min_confidence,
            "tier": self.tier,
            "confab_only": self.confab_only,
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def _pearson(xs: Sequence[float], ys: Sequence[float], *, epsilon: float) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= epsilon or y_var <= epsilon:
        return None
    return numerator / math.sqrt(x_var * y_var)


def _residualize(
    target: Sequence[float],
    control: Sequence[float],
    *,
    epsilon: float,
) -> list[float]:
    if len(target) != len(control) or len(target) < 2:
        return list(target)

    mean_target = sum(target) / len(target)
    mean_control = sum(control) / len(control)
    control_var = sum((value - mean_control) ** 2 for value in control)
    if control_var <= epsilon:
        return [value - mean_target for value in target]

    covariance = sum(
        (control_value - mean_control) * (target_value - mean_target)
        for control_value, target_value in zip(control, target)
    )
    slope = covariance / control_var
    intercept = mean_target - slope * mean_control
    return [value - (intercept + slope * ctrl) for value, ctrl in zip(target, control)]


def _window_final_ratio(window: Sequence[VitalsSnapshot]) -> float:
    last = window[-1]
    if last.source_finding_ratio is not None:
        return float(last.source_finding_ratio)
    return _safe_ratio(last.signals.sources_count, last.signals.findings_count)


def _verified_count(snapshot: VitalsSnapshot) -> int:
    """Safely get verified_sources_count, treating None as 0."""
    return snapshot.signals.verified_sources_count or 0


def _unverified_count(snapshot: VitalsSnapshot) -> int:
    """Safely get unverified_sources_count, treating None as 0."""
    return snapshot.signals.unverified_sources_count or 0


def _has_verified_data(window: Sequence[VitalsSnapshot]) -> bool:
    """Check if the window has meaningful verified source counts."""
    total_verified = sum(_verified_count(s) for s in window)
    total_sources = sum(s.signals.sources_count for s in window)
    return total_sources > 0 and (
        total_verified > 0 or any(_unverified_count(s) > 0 for s in window)
    )


def _score_verified_link(
    window: Sequence[VitalsSnapshot], config: CausalConfabConfig
) -> float | None:
    """Compute verified→sources link strength for a window.

    Measures how well verified source growth tracks total source growth.
    Returns None if verified data is unavailable.
    """
    if not _has_verified_data(window):
        return None

    verified = [_verified_count(s) for s in window]
    sources = [s.signals.sources_count for s in window]
    tokens = [s.signals.total_tokens for s in window]

    verified_deltas = [max(0, c - p) for p, c in zip(verified, verified[1:])]
    sources_deltas = [max(0, c - p) for p, c in zip(sources, sources[1:])]
    token_deltas = [max(0, c - p) for p, c in zip(tokens, tokens[1:])]

    total_sources_delta = sum(sources_deltas)
    if total_sources_delta == 0:
        return None

    verified_response = sum(verified_deltas) / max(1, total_sources_delta)
    verified_residuals = _residualize(verified_deltas, token_deltas, epsilon=config.epsilon)
    sources_residuals = _residualize(sources_deltas, token_deltas, epsilon=config.epsilon)
    corr = _pearson(verified_residuals, sources_residuals, epsilon=config.epsilon)

    if corr is None:
        return min(1.0, verified_response)
    normalized = (corr + 1.0) / 2.0
    return max(0.0, min(1.0, normalized * min(1.0, verified_response)))


def _score_window(window: Sequence[VitalsSnapshot], config: CausalConfabConfig) -> WindowScore:
    findings = [snapshot.signals.findings_count for snapshot in window]
    sources = [snapshot.signals.sources_count for snapshot in window]
    tokens = [snapshot.signals.total_tokens for snapshot in window]

    findings_deltas = [
        max(0, current - previous) for previous, current in zip(findings, findings[1:])
    ]
    sources_deltas = [max(0, current - previous) for previous, current in zip(sources, sources[1:])]
    token_deltas = [max(0, current - previous) for previous, current in zip(tokens, tokens[1:])]

    response_ratio = sum(sources_deltas) / max(1, sum(findings_deltas))
    findings_residuals = _residualize(findings_deltas, token_deltas, epsilon=config.epsilon)
    sources_residuals = _residualize(sources_deltas, token_deltas, epsilon=config.epsilon)
    partial_correlation = _pearson(
        findings_residuals,
        sources_residuals,
        epsilon=config.epsilon,
    )

    if partial_correlation is None:
        link_strength = min(1.0, response_ratio)
    else:
        normalized_corr = (partial_correlation + 1.0) / 2.0
        link_strength = max(0.0, min(1.0, normalized_corr * min(1.0, response_ratio)))

    return WindowScore(
        start_step=window[0].loop_index,
        end_step=window[-1].loop_index,
        link_strength=link_strength,
        response_ratio=response_ratio,
        partial_correlation=partial_correlation,
        source_finding_ratio=_window_final_ratio(window),
        verified_link_strength=_score_verified_link(window, config),
    )


def score_causal_windows(
    snapshots: Sequence[VitalsSnapshot],
    config: CausalConfabConfig | None = None,
) -> tuple[WindowScore, ...]:
    """Return rolling causal-link scores for the supplied trace."""

    effective_config = config or CausalConfabConfig()
    if len(snapshots) < effective_config.window_size:
        return ()

    scores: list[WindowScore] = []
    for end_index in range(effective_config.window_size, len(snapshots) + 1):
        window = snapshots[end_index - effective_config.window_size : end_index]
        scores.append(_score_window(window, effective_config))
    return tuple(scores)


def _result_confidence(
    *,
    config: CausalConfabConfig,
    trigger: str | None,
    baseline_strength: float,
    weakest_strength: float,
    structural_drop: float,
    final_ratio: float,
) -> float:
    if trigger == "causal_link_break":
        baseline_term = min(1.0, baseline_strength / max(config.baseline_floor, config.epsilon))
        drop_term = min(
            1.0,
            structural_drop / max(config.structural_drop_threshold, config.epsilon),
        )
        ratio_term = 1.0 - min(1.0, final_ratio / max(config.ratio_gate, config.epsilon))
        confidence = 0.35 + (0.35 * baseline_term) + (0.2 * drop_term) + (0.1 * ratio_term)
        return min(1.0, confidence)

    if trigger == "persistent_low_causal_link":
        low_link_term = 1.0 - min(
            1.0,
            weakest_strength / max(config.low_link_threshold, config.epsilon),
        )
        ratio_term = 1.0 - min(
            1.0,
            final_ratio / max(config.low_link_ratio_gate, config.epsilon),
        )
        confidence = 0.4 + (0.4 * low_link_term) + (0.2 * ratio_term)
        return min(1.0, confidence)

    drift = max(0.0, baseline_strength - weakest_strength)
    ratio_term = 1.0 - min(1.0, final_ratio / max(config.ratio_gate, config.epsilon))
    return min(0.49, 0.2 + (0.2 * drift) + (0.09 * ratio_term))


def _verified_confidence(
    *,
    config: CausalConfabConfig,
    verified_baseline: float,
    verified_weakest: float,
    verified_drop: float,
    final_ratio: float,
) -> float:
    """Confidence for the verified_source_decoupling trigger."""
    weak_term = 1.0 - min(
        1.0,
        verified_weakest / max(config.verified_weak_threshold, config.epsilon),
    )
    ratio_term = 1.0 - min(
        1.0,
        final_ratio / max(config.verified_ratio_gate, config.epsilon),
    )
    drop_term = (
        min(
            1.0,
            verified_drop / max(config.verified_drop_threshold, config.epsilon),
        )
        if verified_drop > 0
        else 0.0
    )
    return min(1.0, 0.4 + (0.3 * weak_term) + (0.15 * drop_term) + (0.15 * ratio_term))


def detect_causal_confabulation(
    snapshots: Sequence[VitalsSnapshot],
    config: CausalConfabConfig | None = None,
) -> CausalConfabResult:
    """Detect confabulation via a rolling causal source-link score."""

    effective_config = config or CausalConfabConfig()
    final_ratio = _window_final_ratio((snapshots[-1],)) if snapshots else 0.0
    initial_sources = snapshots[0].signals.sources_count if snapshots else 0

    window_scores = score_causal_windows(snapshots, effective_config)
    if not window_scores:
        return CausalConfabResult(
            detected=False,
            confidence=0.0,
            trigger="insufficient_history" if snapshots else None,
            baseline_strength=0.0,
            weakest_strength=0.0,
            structural_drop=0.0,
            final_ratio=final_ratio,
            initial_sources=initial_sources,
            onset_step=None,
            window_scores=window_scores,
        )

    baseline_window = max(window_scores[:2], key=lambda score: score.link_strength)
    comparison_windows = window_scores[1:] or window_scores
    weakest_window = min(comparison_windows, key=lambda score: score.link_strength)
    structural_drop = max(0.0, baseline_window.link_strength - weakest_window.link_strength)

    structural_break = (
        baseline_window.link_strength >= effective_config.baseline_floor
        and weakest_window.link_strength <= effective_config.weak_link_threshold
        and structural_drop >= effective_config.structural_drop_threshold
        and final_ratio <= effective_config.ratio_gate
    )
    persistent_low_link = (
        weakest_window.link_strength <= effective_config.low_link_threshold
        and initial_sources <= effective_config.source_bootstrap_cap
        and final_ratio <= effective_config.low_link_ratio_gate
    )

    # Path 3: verified source decoupling (real LLM confabulation)
    # In elicited traces, findings==sources (1:1), so the primary link never breaks.
    # Instead, verified sources persistently lag behind total sources — every window
    # has a weak verified_link_strength. Detect this persistent weakness.
    has_verified = any(w.verified_link_strength is not None for w in window_scores)
    verified_decoupling = False
    verified_baseline_strength = 0.0
    verified_weakest_strength = 0.0
    verified_drop = 0.0
    verified_weakest_window = weakest_window
    if has_verified:
        verified_scores = [w for w in window_scores if w.verified_link_strength is not None]
        if verified_scores:
            vls_values = [w.verified_link_strength or 0.0 for w in verified_scores]
            verified_baseline_strength = max(vls_values)
            verified_weakest_strength = min(vls_values)
            verified_drop = max(0.0, verified_baseline_strength - verified_weakest_strength)
            verified_weakest_window = min(
                verified_scores, key=lambda w: w.verified_link_strength or 0.0
            )

            total_verified = _verified_count(snapshots[-1])
            total_sources = snapshots[-1].signals.sources_count
            verified_ratio = total_verified / max(1, total_sources)

            # Persistent weak verified link: the STRONGEST window is still weak
            verified_decoupling = (
                verified_baseline_strength <= effective_config.verified_link_floor
                and verified_ratio <= effective_config.verified_ratio_gate
                and total_sources >= effective_config.verified_min_sources
            )

    trigger: str | None
    onset_step: int | None
    if structural_break:
        trigger = "causal_link_break"
        onset_step = weakest_window.start_step
    elif persistent_low_link:
        trigger = "persistent_low_causal_link"
        onset_step = weakest_window.start_step
    elif verified_decoupling:
        trigger = "verified_source_decoupling"
        onset_step = verified_weakest_window.start_step
    else:
        trigger = None
        onset_step = None

    if trigger == "verified_source_decoupling":
        confidence = _verified_confidence(
            config=effective_config,
            verified_baseline=verified_baseline_strength,
            verified_weakest=verified_weakest_strength,
            verified_drop=verified_drop,
            final_ratio=final_ratio,
        )
    else:
        confidence = _result_confidence(
            config=effective_config,
            trigger=trigger,
            baseline_strength=baseline_window.link_strength,
            weakest_strength=weakest_window.link_strength,
            structural_drop=structural_drop,
            final_ratio=final_ratio,
        )

    return CausalConfabResult(
        detected=trigger is not None,
        confidence=confidence,
        trigger=trigger,
        baseline_strength=baseline_window.link_strength,
        weakest_strength=weakest_window.link_strength,
        structural_drop=structural_drop,
        final_ratio=final_ratio,
        initial_sources=initial_sources,
        onset_step=onset_step,
        window_scores=window_scores,
    )


def evaluate_causal_confab_corpus(
    corpus_version: str = "v1",
    *,
    config: CausalConfabConfig | None = None,
    min_confidence: float = 0.8,
    tier: str | None = None,
    confab_only: bool = True,
) -> CausalConfabEvaluation:
    """Evaluate the prototype against the local bench corpus."""

    from evaluator.runner import load_manifest, load_trace

    effective_config = config or CausalConfabConfig()
    manifest = load_manifest(corpus_version)
    metric = DetectorMetrics(detector="causal_confabulation")
    trace_count = 0

    for entry in manifest:
        if entry.get("metadata", {}).get("confidence", 0.0) < min_confidence:
            continue
        if confab_only and not str(entry["path"]).startswith("traces/confabulation/"):
            continue
        if tier is not None and entry.get("tier") != tier:
            continue

        snapshots = load_trace(corpus_version, entry["path"])
        result = detect_causal_confabulation(snapshots, effective_config)
        metric.record(
            predicted=result.detected,
            expected=bool(entry["labels"].get("confabulation", False)),
        )
        trace_count += 1

    return CausalConfabEvaluation(
        corpus_version=corpus_version,
        trace_count=trace_count,
        metric=metric,
        config=effective_config,
        min_confidence=min_confidence,
        tier=tier,
        confab_only=confab_only,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="v1", help="Corpus version to evaluate")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum manifest confidence to include",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Optional manifest tier filter (for example: synthetic or elicited)",
    )
    parser.add_argument(
        "--include-non-confab",
        action="store_true",
        help="Evaluate against the full corpus instead of only confabulation traces",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    evaluation = evaluate_causal_confab_corpus(
        corpus_version=args.corpus,
        min_confidence=args.min_confidence,
        tier=args.tier,
        confab_only=not args.include_non_confab,
    )
    print(json.dumps(evaluation.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
