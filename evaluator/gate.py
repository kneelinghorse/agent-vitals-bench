"""Gate enforcement — evaluate detector promotion using Wilson CI criteria."""

from __future__ import annotations

from typing import Any

from evaluator.metrics import DetectorMetrics

# Gate thresholds (from gate_criteria.md)
MIN_POSITIVES = 25
MIN_PRECISION_LB = 0.80
MIN_RECALL_LB = 0.75


def check_gate(
    metrics: DetectorMetrics,
    *,
    min_positives: int = MIN_POSITIVES,
    min_precision_lb: float = MIN_PRECISION_LB,
    min_recall_lb: float = MIN_RECALL_LB,
) -> dict[str, Any]:
    """Evaluate whether a detector meets HARD GATE promotion criteria.

    Returns a dict with pass/fail status and detailed breakdown.
    """
    p_lb, p_ub = metrics.precision_ci
    r_lb, r_ub = metrics.recall_ci
    total_pos = metrics.total_positives

    has_enough_positives = total_pos >= min_positives
    precision_passes = p_lb >= min_precision_lb
    recall_passes = r_lb >= min_recall_lb

    passed = has_enough_positives and precision_passes and recall_passes

    return {
        "detector": metrics.detector,
        "passed": passed,
        "status": "HARD GATE" if passed else "NO-GO",
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "precision_lb": round(p_lb, 4),
        "recall_lb": round(r_lb, 4),
        "total_positives": total_pos,
        "checks": {
            "min_positives": {
                "required": min_positives,
                "actual": total_pos,
                "passed": has_enough_positives,
            },
            "precision_lb": {
                "required": min_precision_lb,
                "actual": round(p_lb, 4),
                "passed": precision_passes,
            },
            "recall_lb": {
                "required": min_recall_lb,
                "actual": round(r_lb, 4),
                "passed": recall_passes,
            },
        },
    }


def check_all_gates(
    detector_metrics: dict[str, DetectorMetrics],
    *,
    min_positives: int = MIN_POSITIVES,
    min_precision_lb: float = MIN_PRECISION_LB,
    min_recall_lb: float = MIN_RECALL_LB,
) -> dict[str, dict[str, Any]]:
    """Check gate criteria for all detectors."""
    return {
        name: check_gate(
            metrics,
            min_positives=min_positives,
            min_precision_lb=min_precision_lb,
            min_recall_lb=min_recall_lb,
        )
        for name, metrics in detector_metrics.items()
    }
