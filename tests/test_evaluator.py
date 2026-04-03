"""Tests for the evaluator pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
from evaluator.metrics import DetectorMetrics, compute_metrics
from evaluator.gate import check_gate, check_all_gates


class TestDetectorMetrics:
    def test_perfect_precision(self) -> None:
        m = DetectorMetrics(detector="test", tp=10, fp=0, fn=2, tn=8)
        assert m.precision == 1.0

    def test_perfect_recall(self) -> None:
        m = DetectorMetrics(detector="test", tp=10, fp=2, fn=0, tn=8)
        assert m.recall == 1.0

    def test_f1_computation(self) -> None:
        m = DetectorMetrics(detector="test", tp=8, fp=2, fn=2, tn=8)
        assert m.f1 == pytest.approx(0.8, abs=0.01)

    def test_zero_division_safe(self) -> None:
        m = DetectorMetrics(detector="test")
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_wilson_ci_computed(self) -> None:
        m = DetectorMetrics(detector="test", tp=40, fp=2, fn=3, tn=55)
        p_lb, p_ub = m.precision_ci
        assert 0.0 < p_lb < m.precision
        assert m.precision < p_ub <= 1.0

    def test_record_method(self) -> None:
        m = DetectorMetrics(detector="test")
        m.record(predicted=True, expected=True)
        m.record(predicted=True, expected=False)
        m.record(predicted=False, expected=True)
        m.record(predicted=False, expected=False)
        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 1
        assert m.tn == 1


class TestGate:
    def test_passing_gate(self) -> None:
        m = DetectorMetrics(detector="loop", tp=40, fp=1, fn=2, tn=57)
        result = check_gate(m)
        assert result["passed"] is True
        assert result["status"] == "HARD GATE"

    def test_insufficient_positives(self) -> None:
        m = DetectorMetrics(detector="thrash", tp=5, fp=0, fn=0, tn=20)
        result = check_gate(m)
        assert result["passed"] is False
        assert result["checks"]["min_positives"]["passed"] is False

    def test_low_precision_lb(self) -> None:
        # Many false positives drive P_lb below threshold
        m = DetectorMetrics(detector="confab", tp=15, fp=10, fn=1, tn=74)
        result = check_gate(m)
        assert result["checks"]["precision_lb"]["passed"] is False

    def test_check_all_gates(self) -> None:
        metrics = {
            "loop": DetectorMetrics(detector="loop", tp=40, fp=1, fn=1, tn=58),
            "stuck": DetectorMetrics(detector="stuck", tp=35, fp=3, fn=5, tn=57),
        }
        results = check_all_gates(metrics)
        assert "loop" in results
        assert "stuck" in results
