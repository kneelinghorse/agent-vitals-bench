"""Causal runaway-cost prototype based on cost/output decoupling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from agent_vitals.schema import VitalsSnapshot

from evaluator.metrics import DetectorMetrics


@dataclass(frozen=True, slots=True)
class CausalRunawayConfig:
    """Tunable parameters for the runaway-cost prototype."""

    window_size: int = 4
    absolute_cost_floor: float = 2500.0
    cost_growth_multiplier: float = 2.0
    correlation_ceiling: float = 0.1
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3 snapshots")


@dataclass(frozen=True, slots=True)
class RunawayWindowScore:
    """Rolling score for a single cost/output window."""

    start_step: int
    end_step: int
    cost_per_finding: float
    partial_correlation: float
    token_delta_sum: int
    finding_delta_sum: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "cost_per_finding": round(self.cost_per_finding, 4),
            "partial_correlation": round(self.partial_correlation, 4),
            "token_delta_sum": self.token_delta_sum,
            "finding_delta_sum": self.finding_delta_sum,
        }


@dataclass(frozen=True, slots=True)
class CausalRunawayResult:
    """Detection output for a single trace."""

    detected: bool
    confidence: float
    trigger: str | None
    baseline_cost: float
    peak_cost: float
    cost_growth: float
    worst_correlation: float
    onset_step: int | None
    window_scores: tuple[RunawayWindowScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "confidence": round(self.confidence, 4),
            "trigger": self.trigger,
            "baseline_cost": round(self.baseline_cost, 4),
            "peak_cost": round(self.peak_cost, 4),
            "cost_growth": round(self.cost_growth, 4),
            "worst_correlation": round(self.worst_correlation, 4),
            "onset_step": self.onset_step,
            "window_scores": [score.as_dict() for score in self.window_scores],
        }


@dataclass(frozen=True, slots=True)
class CausalRunawayEvaluation:
    """Aggregate corpus-level evaluation for the runaway prototype."""

    corpus_version: str
    trace_count: int
    metric: DetectorMetrics
    config: CausalRunawayConfig
    min_confidence: float
    tier: str | None
    runaway_only: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus_version": self.corpus_version,
            "trace_count": self.trace_count,
            "metric": self.metric.as_dict(),
            "config": asdict(self.config),
            "min_confidence": self.min_confidence,
            "tier": self.tier,
            "runaway_only": self.runaway_only,
        }


@dataclass(frozen=True, slots=True)
class MultiplierSweepResult:
    """Safe-range summary for cost-growth multiplier sweeps."""

    passing_min: float | None
    passing_max: float | None
    passing_width: float
    target_f1: float
    evaluations: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passing_min": self.passing_min,
            "passing_max": self.passing_max,
            "passing_width": round(self.passing_width, 4),
            "target_f1": self.target_f1,
            "evaluations": [
                {"multiplier": round(multiplier, 4), "f1": round(f1_score, 4)}
                for multiplier, f1_score in self.evaluations
            ],
        }


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


def _score_window(
    window: Sequence[VitalsSnapshot], config: CausalRunawayConfig
) -> RunawayWindowScore:
    token_values = [snapshot.signals.total_tokens for snapshot in window]
    finding_values = [snapshot.signals.findings_count for snapshot in window]
    coverage_values = [snapshot.signals.coverage_score for snapshot in window]

    token_deltas = [
        max(0, current - previous) for previous, current in zip(token_values, token_values[1:])
    ]
    finding_deltas = [
        max(0, current - previous) for previous, current in zip(finding_values, finding_values[1:])
    ]
    coverage_deltas = [
        max(0.0, current - previous)
        for previous, current in zip(coverage_values, coverage_values[1:])
    ]

    residual_tokens = _residualize(token_deltas, coverage_deltas, epsilon=config.epsilon)
    residual_findings = _residualize(finding_deltas, coverage_deltas, epsilon=config.epsilon)
    partial_correlation = _pearson(residual_tokens, residual_findings, epsilon=config.epsilon)
    effective_correlation = 0.0 if partial_correlation is None else partial_correlation

    return RunawayWindowScore(
        start_step=window[0].loop_index,
        end_step=window[-1].loop_index,
        cost_per_finding=sum(token_deltas) / max(1, sum(finding_deltas)),
        partial_correlation=effective_correlation,
        token_delta_sum=sum(token_deltas),
        finding_delta_sum=sum(finding_deltas),
    )


def score_runaway_windows(
    snapshots: Sequence[VitalsSnapshot],
    config: CausalRunawayConfig | None = None,
) -> tuple[RunawayWindowScore, ...]:
    """Return rolling runaway scores for the supplied trace."""

    effective_config = config or CausalRunawayConfig()
    if len(snapshots) < effective_config.window_size:
        return ()

    scores: list[RunawayWindowScore] = []
    for end_index in range(effective_config.window_size, len(snapshots) + 1):
        window = snapshots[end_index - effective_config.window_size : end_index]
        scores.append(_score_window(window, effective_config))
    return tuple(scores)


def _result_confidence(
    *,
    config: CausalRunawayConfig,
    trigger: str | None,
    peak_cost: float,
    baseline_cost: float,
    cost_growth: float,
    worst_correlation: float,
) -> float:
    if trigger is None:
        headroom = min(1.0, cost_growth / max(config.cost_growth_multiplier, config.epsilon))
        return min(0.49, 0.15 + (0.2 * headroom))

    growth_term = min(1.0, cost_growth / max(config.cost_growth_multiplier, config.epsilon))
    absolute_cost_term = min(1.0, peak_cost / max(config.absolute_cost_floor, config.epsilon))
    corr_term = 1.0 - min(
        1.0,
        max(worst_correlation, 0.0) / max(config.correlation_ceiling, config.epsilon),
    )
    confidence = 0.35 + (0.3 * growth_term) + (0.2 * absolute_cost_term) + (0.15 * corr_term)
    return min(1.0, confidence)


def detect_causal_runaway_cost(
    snapshots: Sequence[VitalsSnapshot],
    config: CausalRunawayConfig | None = None,
) -> CausalRunawayResult:
    """Detect runaway cost via rolling cost/output decoupling."""

    effective_config = config or CausalRunawayConfig()
    window_scores = score_runaway_windows(snapshots, effective_config)
    if not window_scores:
        return CausalRunawayResult(
            detected=False,
            confidence=0.0,
            trigger="insufficient_history" if snapshots else None,
            baseline_cost=0.0,
            peak_cost=0.0,
            cost_growth=0.0,
            worst_correlation=0.0,
            onset_step=None,
            window_scores=window_scores,
        )

    baseline_window = min(window_scores[:2], key=lambda score: score.cost_per_finding)
    comparison_windows = window_scores[1:] or window_scores
    peak_window = max(comparison_windows, key=lambda score: score.cost_per_finding)
    worst_correlation = min(score.partial_correlation for score in comparison_windows)
    cost_growth = peak_window.cost_per_finding / max(1.0, baseline_window.cost_per_finding)

    detected = (
        peak_window.cost_per_finding >= effective_config.absolute_cost_floor
        and cost_growth >= effective_config.cost_growth_multiplier
        and worst_correlation <= effective_config.correlation_ceiling
    )
    trigger = "cost_output_decoupling" if detected else None
    confidence = _result_confidence(
        config=effective_config,
        trigger=trigger,
        peak_cost=peak_window.cost_per_finding,
        baseline_cost=baseline_window.cost_per_finding,
        cost_growth=cost_growth,
        worst_correlation=worst_correlation,
    )

    return CausalRunawayResult(
        detected=detected,
        confidence=confidence,
        trigger=trigger,
        baseline_cost=baseline_window.cost_per_finding,
        peak_cost=peak_window.cost_per_finding,
        cost_growth=cost_growth,
        worst_correlation=worst_correlation,
        onset_step=peak_window.start_step if detected else None,
        window_scores=window_scores,
    )


def evaluate_causal_runaway_corpus(
    corpus_version: str = "v1",
    *,
    config: CausalRunawayConfig | None = None,
    min_confidence: float = 0.8,
    tier: str | None = None,
    runaway_only: bool = True,
) -> CausalRunawayEvaluation:
    """Evaluate the prototype against the local bench corpus."""

    from evaluator.runner import load_manifest, load_trace

    effective_config = config or CausalRunawayConfig()
    manifest = load_manifest(corpus_version)
    metric = DetectorMetrics(detector="causal_runaway_cost")
    trace_count = 0

    for entry in manifest:
        if entry.get("metadata", {}).get("confidence", 0.0) < min_confidence:
            continue
        if runaway_only and not str(entry["path"]).startswith("traces/runaway_cost/"):
            continue
        if tier is not None and entry.get("tier") != tier:
            continue

        snapshots = load_trace(corpus_version, entry["path"])
        result = detect_causal_runaway_cost(snapshots, effective_config)
        metric.record(
            predicted=result.detected,
            expected=bool(entry["labels"].get("runaway_cost", False)),
        )
        trace_count += 1

    return CausalRunawayEvaluation(
        corpus_version=corpus_version,
        trace_count=trace_count,
        metric=metric,
        config=effective_config,
        min_confidence=min_confidence,
        tier=tier,
        runaway_only=runaway_only,
    )


def sweep_cost_growth_multiplier(
    corpus_version: str = "v1",
    *,
    multipliers: Sequence[float] | None = None,
    config: CausalRunawayConfig | None = None,
    min_confidence: float = 0.8,
    tier: str | None = None,
    runaway_only: bool = True,
    target_f1: float = 0.941,
) -> MultiplierSweepResult:
    """Measure the safe range for the cost-growth multiplier threshold."""

    effective_config = config or CausalRunawayConfig()
    tested_multipliers = multipliers or tuple(step / 10 for step in range(10, 51))
    evaluations: list[tuple[float, float]] = []
    passing: list[float] = []

    for multiplier in tested_multipliers:
        evaluation = evaluate_causal_runaway_corpus(
            corpus_version=corpus_version,
            config=replace(effective_config, cost_growth_multiplier=multiplier),
            min_confidence=min_confidence,
            tier=tier,
            runaway_only=runaway_only,
        )
        f1_score = evaluation.metric.f1
        evaluations.append((multiplier, f1_score))
        if f1_score >= target_f1:
            passing.append(multiplier)

    if not passing:
        passing_min = None
        passing_max = None
        passing_width = 0.0
    else:
        passing_min = passing[0]
        passing_max = passing[-1]
        passing_width = float(passing_max - passing_min)
    return MultiplierSweepResult(
        passing_min=passing_min,
        passing_max=passing_max,
        passing_width=passing_width,
        target_f1=target_f1,
        evaluations=tuple(evaluations),
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
        "--include-non-runaway",
        action="store_true",
        help="Evaluate against the full corpus instead of only runaway traces",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Report the multiplier safe range instead of a single evaluation run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.sweep:
        result = sweep_cost_growth_multiplier(
            corpus_version=args.corpus,
            min_confidence=args.min_confidence,
            tier=args.tier,
            runaway_only=not args.include_non_runaway,
        )
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    evaluation = evaluate_causal_runaway_corpus(
        corpus_version=args.corpus,
        min_confidence=args.min_confidence,
        tier=args.tier,
        runaway_only=not args.include_non_runaway,
    )
    print(json.dumps(evaluation.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
