"""Metrics computation — confusion matrices, precision/recall, Wilson CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_vitals.ci_gate import wilson_interval

if TYPE_CHECKING:
    from evaluator.runner import TraceResult


@dataclass(slots=True)
class DetectorMetrics:
    """Evaluation metrics for a single detector."""

    detector: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def total_positives(self) -> int:
        return self.tp + self.fn

    @property
    def total_predictions(self) -> int:
        return self.tp + self.fp

    @property
    def precision_ci(self) -> tuple[float, float]:
        trials = self.tp + self.fp
        if trials == 0:
            return (0.0, 0.0)
        return wilson_interval(self.tp, trials)

    @property
    def recall_ci(self) -> tuple[float, float]:
        trials = self.tp + self.fn
        if trials == 0:
            return (0.0, 0.0)
        return wilson_interval(self.tp, trials)

    def record(self, *, predicted: bool, expected: bool) -> None:
        if predicted and expected:
            self.tp += 1
        elif predicted and not expected:
            self.fp += 1
        elif not predicted and expected:
            self.fn += 1
        else:
            self.tn += 1

    def as_dict(self) -> dict[str, object]:
        p_lb, p_ub = self.precision_ci
        r_lb, r_ub = self.recall_ci
        return {
            "detector": self.detector,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "precision_lb": round(p_lb, 4),
            "precision_ub": round(p_ub, 4),
            "recall_lb": round(r_lb, 4),
            "recall_ub": round(r_ub, 4),
            "total_positives": self.total_positives,
        }


def compute_metrics(
    trace_results: list[TraceResult],
    detectors: tuple[str, ...],
) -> dict[str, DetectorMetrics]:
    """Compute confusion matrix metrics per detector from trace results."""
    metrics: dict[str, DetectorMetrics] = {d: DetectorMetrics(detector=d) for d in detectors}

    for tr in trace_results:
        for detector in detectors:
            expected = tr.labels.get(detector, False)
            predicted = tr.predictions.get(detector, False)
            metrics[detector].record(predicted=predicted, expected=expected)

    return metrics
