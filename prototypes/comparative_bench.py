"""Comparative benchmark harness for handcrafted, causal, and TDA approaches."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_TDA_PYTHON = REPO_ROOT.parent / "tda-experiment" / ".venv" / "bin" / "python"

DETECTORS: tuple[str, ...] = (
    "loop",
    "stuck",
    "confabulation",
    "thrash",
    "runaway_cost",
)
CAUSAL_DETECTORS: tuple[str, ...] = ("confabulation", "runaway_cost")


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    path: str
    tier: str | None
    confidence: float
    step_count: int
    family: str
    delayed_onset: bool
    labels: dict[str, bool]
    predictions: dict[str, dict[str, bool | None]]
    details: dict[str, dict[str, dict[str, Any]]]

    def get_prediction(self, approach: str, detector: str) -> bool | None:
        return self.predictions.get(approach, {}).get(detector)

    def get_detail(self, approach: str, detector: str) -> dict[str, Any]:
        return self.details.get(approach, {}).get(detector, {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path": self.path,
            "tier": self.tier,
            "confidence": round(self.confidence, 4),
            "step_count": self.step_count,
            "family": self.family,
            "delayed_onset": self.delayed_onset,
            "labels": self.labels,
            "predictions": self.predictions,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    precision_lb: float
    precision_ub: float
    recall_lb: float
    recall_ub: float
    total_positives: int
    total_predictions: int
    evaluated_traces: int

    @classmethod
    def from_counts(
        cls,
        *,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        evaluated_traces: int,
    ) -> MetricSummary:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precision_lb, precision_ub = _wilson_interval(tp, tp + fp)
        recall_lb, recall_ub = _wilson_interval(tp, tp + fn)
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            precision_lb=precision_lb,
            precision_ub=precision_ub,
            recall_lb=recall_lb,
            recall_ub=recall_ub,
            total_positives=tp + fn,
            total_predictions=tp + fp,
            evaluated_traces=evaluated_traces,
        )


@dataclass(frozen=True, slots=True)
class FoldMetricSummary:
    fold_index: int
    trace_count: int
    metric: MetricSummary


@dataclass(frozen=True, slots=True)
class FailureBucket:
    bucket: str
    count: int
    example_trace_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FailureAnalysis:
    false_negatives: int
    false_positives: int
    fn_buckets: tuple[FailureBucket, ...]
    fp_buckets: tuple[FailureBucket, ...]


@dataclass(frozen=True, slots=True)
class PairwiseAgreement:
    left: str
    right: str
    agreement_count: int
    total: int
    agreement_rate: float


@dataclass(frozen=True, slots=True)
class AgreementExample:
    trace_id: str
    path: str
    label: bool
    predictions: dict[str, bool]


@dataclass(frozen=True, slots=True)
class AgreementSummary:
    aligned_trace_count: int
    available_approaches: tuple[str, ...]
    pairwise: tuple[PairwiseAgreement, ...]
    positive_signatures: dict[str, int]
    negative_signatures: dict[str, int]
    disagreement_examples: tuple[AgreementExample, ...]


@dataclass(frozen=True, slots=True)
class Recommendation:
    recommended_approach: str
    rationale: str
    hybrid_option: str | None


@dataclass(frozen=True, slots=True)
class ApproachSummary:
    approach: str
    implemented: bool
    aligned_metric: MetricSummary | None
    full_metric: MetricSummary | None
    fold_metrics: tuple[FoldMetricSummary, ...]
    full_trace_count: int
    aligned_trace_count: int
    coverage_rate: float
    failure_analysis: FailureAnalysis | None


@dataclass(frozen=True, slots=True)
class DetectorComparison:
    detector: str
    approach_summaries: tuple[ApproachSummary, ...]
    agreement: AgreementSummary
    recommendation: Recommendation


@dataclass(frozen=True, slots=True)
class ComparativeBenchmarkResult:
    corpus_version: str
    min_confidence: float
    generated_at: str
    filtered_trace_count: int
    tda_trace_count: int
    tda_excluded_count: int
    tda_excluded_reasons: dict[str, int]
    detectors: tuple[DetectorComparison, ...]
    trace_records: tuple[TraceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "min_confidence": self.min_confidence,
            "generated_at": self.generated_at,
            "filtered_trace_count": self.filtered_trace_count,
            "tda_trace_count": self.tda_trace_count,
            "tda_excluded_count": self.tda_excluded_count,
            "tda_excluded_reasons": self.tda_excluded_reasons,
            "detectors": [asdict(detector) for detector in self.detectors],
            "trace_records": [trace.to_dict() for trace in self.trace_records],
        }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)

    from agent_vitals.ci_gate import wilson_interval

    return wilson_interval(successes, trials)


def classify_trace_family(path: str) -> str:
    parts = set(Path(path).parts)
    if "legacy" in parts:
        return "legacy"
    if "framework" in parts:
        return "framework"
    if "elicited_cross_model" in parts:
        return "elicited_cross_model"
    if "elicited_causal" in parts:
        return "elicited_causal"
    if "elicited" in parts:
        return "elicited"
    return "synthetic"


def _is_delayed_onset(path: str) -> bool:
    lowered = Path(path).name.lower()
    return "delayed" in lowered or "late-onset" in lowered


def collect_trace_records(
    *,
    corpus_version: str,
    min_confidence: float,
) -> list[TraceRecord]:
    from evaluator.runner import load_manifest, load_trace, replay_trace, resolve_workflow_type
    from prototypes.causal_confab import detect_causal_confabulation
    from prototypes.causal_runaway import detect_causal_runaway_cost

    manifest = load_manifest(corpus_version)
    records: list[TraceRecord] = []

    for entry in manifest:
        confidence = float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue

        path = str(entry["path"])
        snapshots = load_trace(corpus_version, path)
        mission_id = snapshots[0].mission_id if snapshots else None
        workflow_type = resolve_workflow_type(str(entry["trace_id"]), mission_id)
        handcrafted = replay_trace(
            snapshots,
            workflow_type=workflow_type,
            runtime_mode="default",
        )

        causal_confab = detect_causal_confabulation(snapshots)
        causal_runaway = detect_causal_runaway_cost(snapshots)

        records.append(
            TraceRecord(
                trace_id=str(entry["trace_id"]),
                path=path,
                tier=entry.get("tier"),
                confidence=confidence,
                step_count=len(snapshots),
                family=classify_trace_family(path),
                delayed_onset=_is_delayed_onset(path),
                labels={
                    detector: bool(entry["labels"].get(detector, False)) for detector in DETECTORS
                },
                predictions={
                    "handcrafted": {
                        detector: bool(handcrafted.get(detector, False)) for detector in DETECTORS
                    },
                    "causal": {
                        "confabulation": causal_confab.detected,
                        "runaway_cost": causal_runaway.detected,
                    },
                    "tda": {},
                },
                details={
                    "causal": {
                        "confabulation": causal_confab.as_dict(),
                        "runaway_cost": causal_runaway.as_dict(),
                    },
                    "tda": {},
                },
            )
        )

    return records


def _build_raw_metric(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    approach: str,
) -> tuple[MetricSummary | None, list[TraceRecord]]:
    evaluated: list[TraceRecord] = []
    tp = fp = fn = tn = 0

    for record in records:
        predicted = record.get_prediction(approach, detector)
        if predicted is None:
            continue
        expected = record.labels.get(detector, False)
        evaluated.append(record)
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    if not evaluated:
        return (None, [])
    return (
        MetricSummary.from_counts(tp=tp, fp=fp, fn=fn, tn=tn, evaluated_traces=len(evaluated)),
        evaluated,
    )


def _stratified_folds(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    n_splits: int,
    seed: int,
) -> list[list[TraceRecord]]:
    positives = [record for record in records if record.labels.get(detector, False)]
    negatives = [record for record in records if not record.labels.get(detector, False)]

    rng = random.Random(f"{seed}:{detector}")
    rng.shuffle(positives)
    rng.shuffle(negatives)

    folds: list[list[TraceRecord]] = [[] for _ in range(n_splits)]
    for index, record in enumerate(positives):
        folds[index % n_splits].append(record)
    for index, record in enumerate(negatives):
        folds[index % n_splits].append(record)
    return folds


def build_fold_metrics(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    approach: str,
    n_splits: int,
    seed: int,
) -> tuple[FoldMetricSummary, ...]:
    metric, evaluated_records = _build_raw_metric(records, detector=detector, approach=approach)
    if metric is None or len(evaluated_records) < n_splits:
        return ()

    fold_metrics: list[FoldMetricSummary] = []
    for fold_index, fold_records in enumerate(
        _stratified_folds(evaluated_records, detector=detector, n_splits=n_splits, seed=seed),
        start=1,
    ):
        fold_metric, _ = _build_raw_metric(fold_records, detector=detector, approach=approach)
        if fold_metric is None:
            continue
        fold_metrics.append(
            FoldMetricSummary(
                fold_index=fold_index,
                trace_count=len(fold_records),
                metric=fold_metric,
            )
        )
    return tuple(fold_metrics)


def _fold_metrics_from_payload(
    payload: Sequence[dict[str, int]],
) -> tuple[FoldMetricSummary, ...]:
    fold_metrics: list[FoldMetricSummary] = []
    for item in payload:
        fold_metrics.append(
            FoldMetricSummary(
                fold_index=int(item["fold_index"]),
                trace_count=int(item["trace_count"]),
                metric=MetricSummary.from_counts(
                    tp=int(item["tp"]),
                    fp=int(item["fp"]),
                    fn=int(item["fn"]),
                    tn=int(item["tn"]),
                    evaluated_traces=int(item["trace_count"]),
                ),
            )
        )
    return tuple(fold_metrics)


def _causal_confab_reason(detail: dict[str, Any]) -> str:
    trigger = detail.get("trigger")
    if trigger == "insufficient_history":
        return "insufficient_history"
    baseline = float(detail.get("baseline_strength", 0.0) or 0.0)
    weakest = float(detail.get("weakest_strength", 0.0) or 0.0)
    structural_drop = float(detail.get("structural_drop", 0.0) or 0.0)
    final_ratio = float(detail.get("final_ratio", 0.0) or 0.0)

    if baseline < 0.4:
        return "baseline_never_established"
    if weakest > 0.35:
        return "link_never_weakened"
    if structural_drop < 0.2:
        return "structural_drop_too_small"
    if final_ratio > 0.4:
        return "source_ratio_recovered"
    return "no_trigger"


def _causal_runaway_reason(detail: dict[str, Any]) -> str:
    trigger = detail.get("trigger")
    if trigger == "insufficient_history":
        return "insufficient_history"
    peak_cost = float(detail.get("peak_cost", 0.0) or 0.0)
    cost_growth = float(detail.get("cost_growth", 0.0) or 0.0)
    worst_correlation = float(detail.get("worst_correlation", 0.0) or 0.0)

    if peak_cost < 2500.0:
        return "peak_cost_below_floor"
    if cost_growth < 2.0:
        return "cost_growth_too_small"
    if worst_correlation > 0.1:
        return "cost_still_tracks_findings"
    return "no_trigger"


def _failure_bucket(
    record: TraceRecord,
    *,
    detector: str,
    approach: str,
    error_type: str,
) -> str:
    if error_type == "fp":
        trigger = record.get_detail(approach, detector).get("trigger")
        if isinstance(trigger, str) and trigger:
            base = trigger
        elif record.delayed_onset:
            base = "delayed_onset"
        else:
            base = record.family
    elif approach == "causal" and detector == "confabulation":
        base = _causal_confab_reason(record.get_detail("causal", "confabulation"))
    elif approach == "causal" and detector == "runaway_cost":
        base = _causal_runaway_reason(record.get_detail("causal", "runaway_cost"))
    elif record.delayed_onset:
        base = "delayed_onset"
    else:
        base = record.family

    if record.delayed_onset and base != "delayed_onset":
        return f"{base}+delayed_onset"
    return base


def summarize_failures(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    approach: str,
    max_examples: int = 5,
) -> FailureAnalysis | None:
    metric, evaluated = _build_raw_metric(records, detector=detector, approach=approach)
    if metric is None:
        return None

    false_negatives = [
        record
        for record in evaluated
        if not bool(record.get_prediction(approach, detector))
        and record.labels.get(detector, False)
    ]
    false_positives = [
        record
        for record in evaluated
        if bool(record.get_prediction(approach, detector))
        and not record.labels.get(detector, False)
    ]

    def bucketize(
        items: Sequence[TraceRecord],
        *,
        error_type: str,
    ) -> tuple[FailureBucket, ...]:
        grouped: dict[str, list[str]] = {}
        for record in items:
            grouped.setdefault(
                _failure_bucket(
                    record,
                    detector=detector,
                    approach=approach,
                    error_type=error_type,
                ),
                [],
            ).append(record.trace_id)

        ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        return tuple(
            FailureBucket(
                bucket=bucket,
                count=len(trace_ids),
                example_trace_ids=tuple(trace_ids[:max_examples]),
            )
            for bucket, trace_ids in ordered
        )

    return FailureAnalysis(
        false_negatives=len(false_negatives),
        false_positives=len(false_positives),
        fn_buckets=bucketize(false_negatives, error_type="fn"),
        fp_buckets=bucketize(false_positives, error_type="fp"),
    )


def _prediction_signature(
    record: TraceRecord,
    *,
    detector: str,
    approaches: Sequence[str],
) -> str:
    positives = [
        approach for approach in approaches if bool(record.get_prediction(approach, detector))
    ]
    if positives:
        return "+".join(positives)
    return "none"


def build_agreement_summary(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    approaches: Sequence[str],
    max_examples: int = 6,
) -> AgreementSummary:
    aligned = [
        record
        for record in records
        if all(record.get_prediction(approach, detector) is not None for approach in approaches)
    ]
    pairwise: list[PairwiseAgreement] = []
    for index, left in enumerate(approaches):
        for right in approaches[index + 1 :]:
            agreement_count = sum(
                1
                for record in aligned
                if record.get_prediction(left, detector) == record.get_prediction(right, detector)
            )
            total = len(aligned)
            pairwise.append(
                PairwiseAgreement(
                    left=left,
                    right=right,
                    agreement_count=agreement_count,
                    total=total,
                    agreement_rate=(agreement_count / total) if total else 0.0,
                )
            )

    positive_signatures = Counter(
        _prediction_signature(record, detector=detector, approaches=approaches)
        for record in aligned
        if record.labels.get(detector, False)
    )
    negative_signatures = Counter(
        _prediction_signature(record, detector=detector, approaches=approaches)
        for record in aligned
        if not record.labels.get(detector, False)
    )

    disagreements = [
        AgreementExample(
            trace_id=record.trace_id,
            path=record.path,
            label=record.labels.get(detector, False),
            predictions={
                approach: bool(record.get_prediction(approach, detector)) for approach in approaches
            },
        )
        for record in aligned
        if len({record.get_prediction(approach, detector) for approach in approaches}) > 1
    ]

    return AgreementSummary(
        aligned_trace_count=len(aligned),
        available_approaches=tuple(approaches),
        pairwise=tuple(pairwise),
        positive_signatures=dict(positive_signatures),
        negative_signatures=dict(negative_signatures),
        disagreement_examples=tuple(disagreements[:max_examples]),
    )


def _approach_summary(
    records: Sequence[TraceRecord],
    *,
    detector: str,
    approach: str,
    aligned_records: Sequence[TraceRecord],
    full_filtered_count: int,
    n_splits: int,
    seed: int,
    tda_fold_metrics: Sequence[dict[str, int]] | None = None,
) -> ApproachSummary:
    implemented = any(record.get_prediction(approach, detector) is not None for record in records)
    if not implemented:
        return ApproachSummary(
            approach=approach,
            implemented=False,
            aligned_metric=None,
            full_metric=None,
            fold_metrics=(),
            full_trace_count=0,
            aligned_trace_count=0,
            coverage_rate=0.0,
            failure_analysis=None,
        )

    full_metric, full_evaluated = _build_raw_metric(records, detector=detector, approach=approach)
    aligned_metric, aligned_evaluated = _build_raw_metric(
        aligned_records, detector=detector, approach=approach
    )
    if approach == "tda" and tda_fold_metrics is not None:
        fold_metrics = _fold_metrics_from_payload(tda_fold_metrics)
    else:
        fold_metrics = build_fold_metrics(
            aligned_records,
            detector=detector,
            approach=approach,
            n_splits=n_splits,
            seed=seed,
        )
    failure_analysis = summarize_failures(
        aligned_records,
        detector=detector,
        approach=approach,
    )

    full_trace_count = len(full_evaluated)
    aligned_trace_count = len(aligned_evaluated)
    return ApproachSummary(
        approach=approach,
        implemented=True,
        aligned_metric=aligned_metric,
        full_metric=full_metric,
        fold_metrics=fold_metrics,
        full_trace_count=full_trace_count,
        aligned_trace_count=aligned_trace_count,
        coverage_rate=(full_trace_count / full_filtered_count) if full_filtered_count else 0.0,
        failure_analysis=failure_analysis,
    )


def _recommendation_for_detector(
    detector: str,
    approach_summaries: Sequence[ApproachSummary],
) -> Recommendation:
    implemented = [
        summary
        for summary in approach_summaries
        if summary.implemented and summary.aligned_metric is not None
    ]
    if not implemented:
        return Recommendation(
            recommended_approach="none",
            rationale="No approach produced comparable aligned metrics for this detector.",
            hybrid_option=None,
        )

    best = max(
        implemented,
        key=lambda summary: (
            summary.aligned_metric.f1 if summary.aligned_metric is not None else -1.0,
            summary.aligned_metric.recall_lb if summary.aligned_metric is not None else -1.0,
            summary.aligned_metric.precision_lb if summary.aligned_metric is not None else -1.0,
        ),
    )
    handcrafted = next(
        (summary for summary in implemented if summary.approach == "handcrafted"), None
    )

    if (
        handcrafted is not None
        and handcrafted.aligned_metric is not None
        and best.aligned_metric is not None
        and detector in {"loop", "stuck", "thrash"}
        and handcrafted.aligned_metric.f1 >= best.aligned_metric.f1 - 0.01
        and handcrafted.aligned_metric.recall_lb >= best.aligned_metric.recall_lb - 0.02
        and handcrafted.aligned_metric.precision_lb >= best.aligned_metric.precision_lb - 0.02
    ):
        return Recommendation(
            recommended_approach="handcrafted",
            rationale=(
                "Handcrafted remains effectively tied with the best alternative on the aligned "
                "subset while avoiding training and optional TDA dependencies."
            ),
            hybrid_option="Keep TDA as an offline audit path rather than the primary detector.",
        )

    if detector == "confabulation" and best.approach == "tda":
        return Recommendation(
            recommended_approach="tda",
            rationale=(
                "TDA wins the aligned confabulation comparison, which matters because "
                "confabulation is the most fragile handcrafted detector."
            ),
            hybrid_option=(
                "Use the causal detector as an interpretable guardrail or diagnostic fallback "
                "when explaining why a confab trace was flagged."
            ),
        )

    if detector == "runaway_cost" and best.approach in {"tda", "causal"}:
        return Recommendation(
            recommended_approach=best.approach,
            rationale=(
                "The selected approach improves the runaway-cost lower bounds on the aligned "
                "subset, which is the key weakness of the handcrafted threshold path."
            ),
            hybrid_option=(
                "Retain handcrafted burn-rate checks as a cheap first-pass screen and use the "
                f"{best.approach} path as the adjudication layer."
            ),
        )

    if best.approach == "handcrafted":
        rationale = (
            "Handcrafted posts the strongest aligned metrics on this detector and already matches "
            "production semantics."
        )
    else:
        rationale = (
            f"{best.approach.capitalize()} posts the strongest aligned metrics on this detector "
            "and is the best candidate for upstreaming if the added complexity is acceptable."
        )
    return Recommendation(
        recommended_approach=best.approach,
        rationale=rationale,
        hybrid_option=None,
    )


def _integrate_tda_predictions(
    records: Sequence[TraceRecord],
    tda_payload: dict[str, Any],
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    predictions_by_detector = tda_payload["predictions"]
    trace_ids = set(tda_payload["trace_ids"])

    aligned: list[TraceRecord] = []
    full_records = list(records)
    for record in full_records:
        if record.trace_id in trace_ids:
            for detector, predictions in predictions_by_detector.items():
                trace_prediction = predictions.get(record.trace_id)
                if trace_prediction is None:
                    continue
                record.predictions.setdefault("tda", {})[detector] = bool(
                    trace_prediction["detected"]
                )
                record.details.setdefault("tda", {})[detector] = {
                    "probability": float(trace_prediction["probability"])
                }
            aligned.append(record)
    return full_records, aligned


def run_tda_subprocess(
    *,
    corpus_version: str,
    min_confidence: float,
    tda_python: Path,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    request = {
        "corpus_root": str(REPO_ROOT / "corpus" / corpus_version),
        "min_confidence": min_confidence,
        "detectors": list(DETECTORS),
        "n_splits": n_splits,
        "seed": seed,
    }
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".strip(":"),
        }
    )
    completed = subprocess.run(
        [str(tda_python), "-m", "prototypes.comparative_bench", "tda-cv"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TDA cross-validation subprocess failed:\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("tda-cv subprocess returned a non-object JSON payload")
    return payload


def build_comparative_result(
    *,
    corpus_version: str,
    min_confidence: float,
    tda_python: Path,
    n_splits: int = 5,
    seed: int = 42,
) -> ComparativeBenchmarkResult:
    records = collect_trace_records(corpus_version=corpus_version, min_confidence=min_confidence)
    tda_payload = run_tda_subprocess(
        corpus_version=corpus_version,
        min_confidence=min_confidence,
        tda_python=tda_python,
        n_splits=n_splits,
        seed=seed,
    )
    full_records, aligned_records = _integrate_tda_predictions(records, tda_payload)

    detectors: list[DetectorComparison] = []
    for detector in DETECTORS:
        approaches = ["handcrafted", "tda"]
        if detector in CAUSAL_DETECTORS:
            approaches.insert(1, "causal")

        approach_summaries = tuple(
            _approach_summary(
                full_records,
                detector=detector,
                approach=approach,
                aligned_records=aligned_records,
                full_filtered_count=len(full_records),
                n_splits=n_splits,
                seed=seed,
                tda_fold_metrics=tda_payload.get("fold_metrics", {}).get(detector)
                if approach == "tda"
                else None,
            )
            for approach in approaches
        )
        agreement = build_agreement_summary(
            aligned_records,
            detector=detector,
            approaches=approaches,
        )
        recommendation = _recommendation_for_detector(detector, approach_summaries)
        detectors.append(
            DetectorComparison(
                detector=detector,
                approach_summaries=approach_summaries,
                agreement=agreement,
                recommendation=recommendation,
            )
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    excluded_reasons = Counter(
        reason["reason"] for reason in tda_payload.get("excluded_traces", []) if "reason" in reason
    )

    return ComparativeBenchmarkResult(
        corpus_version=corpus_version,
        min_confidence=min_confidence,
        generated_at=generated_at,
        filtered_trace_count=len(full_records),
        tda_trace_count=int(tda_payload["trace_count"]),
        tda_excluded_count=len(tda_payload.get("excluded_traces", [])),
        tda_excluded_reasons=dict(excluded_reasons),
        detectors=tuple(detectors),
        trace_records=tuple(full_records),
    )


def _format_metric(metric: MetricSummary | None, attribute: str) -> str:
    if metric is None:
        return "N/A"
    value = getattr(metric, attribute)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown_report(result: ComparativeBenchmarkResult) -> str:
    lines: list[str] = []
    lines.append("# Causal vs TDA Comparative Benchmark")
    lines.append("")
    lines.append(f"**Generated:** {result.generated_at}")
    lines.append(f"**Corpus:** {result.corpus_version}")
    lines.append(f"**Min confidence:** {result.min_confidence:.1f}")
    lines.append(f"**Filtered traces:** {result.filtered_trace_count}")
    lines.append(f"**TDA-eligible traces:** {result.tda_trace_count}")
    lines.append(f"**TDA exclusions:** {result.tda_excluded_count}")
    lines.append("")
    lines.append("## Coverage and Cross-Validation Setup")
    lines.append("")
    lines.append("- Handcrafted and causal predictions are scored locally on the bench corpus.")
    lines.append(
        "- TDA uses 5-fold stratified cross-validation through the sibling Python 3.12 "
        "`tda-experiment` environment."
    )
    lines.append(
        "- Comparative metrics, agreement matrices, and failure analysis use the "
        "TDA-eligible subset so every approach is compared on the same traces."
    )
    if result.tda_excluded_reasons:
        reason_text = ", ".join(
            f"{reason}={count}" for reason, count in sorted(result.tda_excluded_reasons.items())
        )
        lines.append(f"- TDA exclusions by reason: {reason_text}")
    lines.append("")
    lines.append("## Per-Detector Comparison")
    lines.append("")
    lines.append("| Detector | Approach | Coverage | F1 | P_lb | R_lb | TP | FP | FN |")
    lines.append("|----------|----------|----------|----|------|------|----|----|----|")
    for detector_result in result.detectors:
        for summary in detector_result.approach_summaries:
            metric = summary.aligned_metric
            coverage = f"{summary.full_trace_count}/{result.filtered_trace_count}"
            lines.append(
                f"| {detector_result.detector} | {summary.approach} | {coverage} | "
                f"{_format_metric(metric, 'f1')} | {_format_metric(metric, 'precision_lb')} | "
                f"{_format_metric(metric, 'recall_lb')} | {_format_metric(metric, 'tp')} | "
                f"{_format_metric(metric, 'fp')} | {_format_metric(metric, 'fn')} |"
            )
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("| Detector | Recommendation | Why | Hybrid Option |")
    lines.append("|----------|----------------|-----|---------------|")
    for detector_result in result.detectors:
        recommendation = detector_result.recommendation
        hybrid = recommendation.hybrid_option or "None"
        lines.append(
            f"| {detector_result.detector} | {recommendation.recommended_approach} | "
            f"{recommendation.rationale} | {hybrid} |"
        )
    lines.append("")
    lines.append("## Agreement and Divergence")
    lines.append("")
    for detector_result in result.detectors:
        agreement = detector_result.agreement
        lines.append(f"### {detector_result.detector}")
        lines.append(f"- Aligned traces: {agreement.aligned_trace_count}")
        for pairwise in agreement.pairwise:
            lines.append(
                f"- Pairwise agreement {pairwise.left} vs {pairwise.right}: "
                f"{pairwise.agreement_rate:.3f} ({pairwise.agreement_count}/{pairwise.total})"
            )
        if agreement.positive_signatures:
            top_positive = ", ".join(
                f"{signature}={count}"
                for signature, count in sorted(
                    agreement.positive_signatures.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:5]
            )
            lines.append(f"- Positive-trace signatures: {top_positive}")
        if agreement.negative_signatures:
            top_negative = ", ".join(
                f"{signature}={count}"
                for signature, count in sorted(
                    agreement.negative_signatures.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:5]
            )
            lines.append(f"- Negative-trace signatures: {top_negative}")
        for example in agreement.disagreement_examples:
            prediction_text = ", ".join(
                f"{approach}={value}" for approach, value in example.predictions.items()
            )
            lines.append(
                f"- Divergence example `{example.trace_id}` ({example.path}, label={example.label}): "
                f"{prediction_text}"
            )
        lines.append("")
    lines.append("## Failure Mode Analysis")
    lines.append("")
    for detector_result in result.detectors:
        lines.append(f"### {detector_result.detector}")
        for summary in detector_result.approach_summaries:
            if not summary.implemented or summary.failure_analysis is None:
                lines.append(f"- {summary.approach}: not implemented for this detector")
                continue
            failure = summary.failure_analysis
            lines.append(
                f"- {summary.approach}: FN={failure.false_negatives}, FP={failure.false_positives}"
            )
            if failure.fn_buckets:
                bucket_text = ", ".join(
                    f"{bucket.bucket}={bucket.count}" for bucket in failure.fn_buckets[:4]
                )
                lines.append(f"- {summary.approach} FN buckets: {bucket_text}")
            if failure.fp_buckets:
                bucket_text = ", ".join(
                    f"{bucket.bucket}={bucket.count}" for bucket in failure.fp_buckets[:4]
                )
                lines.append(f"- {summary.approach} FP buckets: {bucket_text}")
        lines.append("")
    return "\n".join(lines)


def save_report(
    result: ComparativeBenchmarkResult,
    *,
    markdown_path: Path,
    json_path: Path,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown_report(result) + "\n")
    json_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")


def _load_raw_steps(trace_path: Path) -> list[dict[str, Any]] | None:
    content = trace_path.read_text().strip()
    if not content:
        return None
    if content.startswith("["):
        payload = json.loads(content)
        return payload if isinstance(payload, list) else None

    steps: list[dict[str, Any]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            return None
        steps.append(parsed)
    return steps


def _tda_cv_subcommand() -> int:
    payload = json.loads(sys.stdin.read())
    if not isinstance(payload, dict):
        raise ValueError("tda-cv expects a JSON object on stdin")

    corpus_root = Path(str(payload["corpus_root"]))
    min_confidence = float(payload.get("min_confidence", 0.8))
    detectors = tuple(str(detector) for detector in payload.get("detectors", DETECTORS))
    n_splits = int(payload.get("n_splits", 5))
    seed = int(payload.get("seed", 42))

    from prototypes.tda_detector import (
        TDAConfig,
        extract_tda_features,
        load_manifest_entries,
        _load_tda_backend,
    )

    config = TDAConfig(
        corpus_root=corpus_root,
        min_confidence=min_confidence,
        n_splits=n_splits,
        random_state=seed,
    )
    manifest_entries = load_manifest_entries(config=config)

    features: list[list[float]] = []
    labels: list[dict[str, bool]] = []
    trace_ids: list[str] = []
    excluded_traces: list[dict[str, Any]] = []

    for entry in manifest_entries:
        trace_id = str(entry["trace_id"])
        trace_path = corpus_root / str(entry["path"])
        if not trace_path.exists():
            excluded_traces.append(
                {"trace_id": trace_id, "path": str(entry["path"]), "reason": "missing_trace"}
            )
            continue
        steps = _load_raw_steps(trace_path)
        if steps is None:
            excluded_traces.append(
                {"trace_id": trace_id, "path": str(entry["path"]), "reason": "invalid_trace_format"}
            )
            continue
        if len(steps) < config.min_steps:
            excluded_traces.append(
                {"trace_id": trace_id, "path": str(entry["path"]), "reason": "insufficient_steps"}
            )
            continue
        vector = extract_tda_features(steps, config=config)
        if vector is None:
            short_trace_cutoff = max(config.window_sizes) + 2
            reason = (
                "short_trace_dropout"
                if len(steps) < short_trace_cutoff
                else "feature_extraction_failed"
            )
            excluded_traces.append(
                {
                    "trace_id": trace_id,
                    "path": str(entry["path"]),
                    "reason": reason,
                }
            )
            continue

        features.append(vector)
        labels.append(
            {detector: bool(entry["labels"].get(detector, False)) for detector in DETECTORS}
        )
        trace_ids.append(trace_id)

    if not features:
        raise ValueError("No traces produced usable TDA features")

    backend = _load_tda_backend()
    np = backend["np"]
    X = np.array(features)

    predictions: dict[str, dict[str, dict[str, float | bool]]] = {}
    metrics: dict[str, dict[str, int]] = {}
    fold_metrics: dict[str, list[dict[str, int]]] = {}

    for detector in detectors:
        y = np.array([1 if label.get(detector, False) else 0 for label in labels], dtype=int)
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        split_count = min(n_splits, positives, negatives)
        if split_count < 2:
            continue

        pipeline = backend["Pipeline"](
            [
                ("scale", backend["StandardScaler"]()),
                (
                    "clf",
                    backend["GradientBoostingClassifier"](
                        n_estimators=config.n_estimators,
                        max_depth=config.max_depth,
                        learning_rate=config.learning_rate,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
        cv = backend["StratifiedKFold"](
            n_splits=split_count,
            shuffle=True,
            random_state=config.random_state,
        )
        y_pred = backend["cross_val_predict"](pipeline, X, y, cv=cv)
        y_proba = backend["cross_val_predict"](pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        tn, fp, fn, tp = backend["confusion_matrix"](y, y_pred, labels=[0, 1]).ravel()
        metrics[detector] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "evaluated_traces": int(len(y)),
        }
        predictions[detector] = {
            trace_id: {
                "detected": bool(y_pred[index]),
                "probability": float(y_proba[index]),
            }
            for index, trace_id in enumerate(trace_ids)
        }

        detector_fold_metrics: list[dict[str, int]] = []
        cv_for_metrics = backend["StratifiedKFold"](
            n_splits=split_count,
            shuffle=True,
            random_state=config.random_state,
        )
        for fold_index, (_, test_indices) in enumerate(cv_for_metrics.split(X, y), start=1):
            y_true_fold = y[test_indices]
            y_pred_fold = y_pred[test_indices]
            fold_tn, fold_fp, fold_fn, fold_tp = backend["confusion_matrix"](
                y_true_fold,
                y_pred_fold,
                labels=[0, 1],
            ).ravel()
            detector_fold_metrics.append(
                {
                    "fold_index": fold_index,
                    "trace_count": int(len(test_indices)),
                    "tp": int(fold_tp),
                    "fp": int(fold_fp),
                    "fn": int(fold_fn),
                    "tn": int(fold_tn),
                }
            )
        fold_metrics[detector] = detector_fold_metrics

    sys.stdout.write(
        json.dumps(
            {
                "trace_count": len(trace_ids),
                "trace_ids": trace_ids,
                "excluded_traces": excluded_traces,
                "metrics": metrics,
                "fold_metrics": fold_metrics,
                "predictions": predictions,
            }
        )
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("tda-cv", help="Internal TDA cross-validation entrypoint")

    parser.add_argument("--corpus", default="v1", help="Corpus version to compare")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum manifest confidence to include",
    )
    parser.add_argument(
        "--tda-python",
        default=str(DEFAULT_TDA_PYTHON),
        help="Python interpreter with numpy/sklearn/gtda installed",
    )
    parser.add_argument(
        "--markdown-output",
        default=str(REPORTS_DIR / "eval-causal-tda-comparison.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--json-output",
        default=str(REPORTS_DIR / "eval-causal-tda-comparison.json"),
        help="JSON report output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "tda-cv":
        return _tda_cv_subcommand()

    result = build_comparative_result(
        corpus_version=args.corpus,
        min_confidence=args.min_confidence,
        tda_python=Path(args.tda_python),
    )
    save_report(
        result,
        markdown_path=Path(args.markdown_output),
        json_path=Path(args.json_output),
    )
    print(render_markdown_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
