"""Tests for the five-paradigm TracePredictor adapters and report orchestration.

The learned-paradigm adapters (TDA / Mamba / Hopfield) need optional ML
dependencies (torch / gtda / hflayers) and trained model artifacts. These
tests run from bench's main Python 3.14 venv where those deps are absent,
so tests for those adapters use ``pytest.importorskip`` and are skipped
cleanly when the deps are missing.

Handcrafted and causal adapters have no heavy deps and are tested in full.
The orchestration helpers (CellResult, ComparativeReport, render_markdown)
are tested with synthetic data so they don't need any of the prototypes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from evaluator.metrics import DetectorMetrics
from evaluator.partial_trace import CutoffMetrics
from evaluator.runner import DETECTORS as RUNNER_DETECTORS
from prototypes.predictor_adapters import (
    DETECTORS,
    PARADIGMS,
    HOPFIELD_PREFIX_FOR_CUTOFF,
    build_predictor,
    make_causal_predictor,
    make_handcrafted_predictor,
)


# ============================================================
# Fixtures
# ============================================================


def _make_snapshot(loop_index: int) -> VitalsSnapshot:
    return VitalsSnapshot(
        spec_version="1.0.0",
        mission_id="test-mission",
        run_id="test-run",
        thread_id=None,
        loop_index=loop_index,
        signals=RawSignals(
            findings_count=loop_index + 1,
            sources_count=loop_index + 1,
            objectives_covered=1,
            coverage_score=0.5,
            confidence_score=0.7,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300 * (loop_index + 1),
            api_calls=0,
            query_count=1,
            unique_domains=1,
            refinement_count=0,
            verified_sources_count=loop_index + 1,
            unverified_sources_count=0,
            convergence_delta=0.05,
            error_count=0,
        ),
        metrics=TemporalMetricsResult(
            cv_coverage=0.0,
            cv_findings_rate=0.0,
            dm_coverage=0.5,
            dm_findings=0.5,
            qpf_tokens=0.5,
            cs_effort=0.5,
        ),
        health_state="healthy",
    )


def _make_trace(steps: int) -> list[VitalsSnapshot]:
    return [_make_snapshot(i) for i in range(steps)]


# ============================================================
# Module-level invariants
# ============================================================


def test_paradigms_canonical_set() -> None:
    """The five paradigms are exactly the canonical set."""
    assert PARADIGMS == ("handcrafted", "causal", "tda", "mamba", "hopfield")


def test_detectors_match_evaluator_runner() -> None:
    """The adapter module re-exports the runner's canonical DETECTORS."""
    assert DETECTORS == RUNNER_DETECTORS


def test_hopfield_prefix_map_covers_default_cutoffs() -> None:
    """Each default cutoff has a corresponding Hopfield prefix entry."""
    assert HOPFIELD_PREFIX_FOR_CUTOFF[3] == 3
    assert HOPFIELD_PREFIX_FOR_CUTOFF[5] == 5
    assert HOPFIELD_PREFIX_FOR_CUTOFF[7] == 7
    assert HOPFIELD_PREFIX_FOR_CUTOFF[None] is None


# ============================================================
# Handcrafted adapter
# ============================================================


def test_handcrafted_predictor_returns_full_detector_dict() -> None:
    predictor = make_handcrafted_predictor()
    result = predictor(_make_trace(5))
    assert set(result.keys()) >= set(DETECTORS)
    for value in result.values():
        assert isinstance(value, bool)


def test_handcrafted_predictor_handles_empty_trace() -> None:
    predictor = make_handcrafted_predictor()
    result = predictor([])
    assert all(value is False for value in result.values())


# ============================================================
# Causal adapter
# ============================================================


def test_causal_predictor_returns_all_five_keys() -> None:
    """Causal only computes confab+runaway, but emits keys for all 5 detectors."""
    predictor = make_causal_predictor()
    result = predictor(_make_trace(8))
    assert set(result.keys()) == set(DETECTORS)
    # Loop / stuck / thrash always default to False under standalone semantics
    assert result["loop"] is False
    assert result["stuck"] is False
    assert result["thrash"] is False


def test_causal_predictor_handles_empty_trace() -> None:
    predictor = make_causal_predictor()
    result = predictor([])
    assert all(value is False for value in result.values())


def test_causal_predictor_short_trace_returns_negatives() -> None:
    """Below the rolling causal window the prototypes return ``insufficient_history``."""
    predictor = make_causal_predictor()
    result = predictor(_make_trace(2))
    assert result["confabulation"] is False
    assert result["runaway_cost"] is False


# ============================================================
# Build dispatch
# ============================================================


def test_build_predictor_handcrafted() -> None:
    predictor = build_predictor("handcrafted")
    assert callable(predictor)
    result = predictor(_make_trace(5))
    assert set(result.keys()) >= set(DETECTORS)


def test_build_predictor_causal() -> None:
    predictor = build_predictor("causal")
    assert callable(predictor)
    result = predictor(_make_trace(5))
    assert set(result.keys()) == set(DETECTORS)


def test_build_predictor_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown paradigm"):
        build_predictor("nonexistent")


def test_build_predictor_hopfield_dispatch_passes_cutoff() -> None:
    """Hopfield dispatch maps cutoff → prefix_len without needing the model files.

    We don't actually load the model artifacts here (which would require
    torch); we just verify that ``build_predictor`` doesn't raise on cutoff
    lookup. If torch is missing the call will fail at the import line, which
    is the correct failure mode — the caller is responsible for installing
    the heavy deps before running the learned paradigms.
    """
    pytest.importorskip("torch")
    pytest.importorskip("hflayers")
    # If we got past importorskip, the build itself should succeed.
    predictor = build_predictor("hopfield", cutoff=3)
    assert callable(predictor)


# ============================================================
# Learned-paradigm adapters (skipped when deps missing)
# ============================================================


def test_make_tda_predictor_skips_without_gtda() -> None:
    pytest.importorskip("gtda")
    from prototypes.predictor_adapters import make_tda_predictor

    predictor = make_tda_predictor()
    assert callable(predictor)


def test_make_mamba_predictor_skips_without_torch() -> None:
    pytest.importorskip("torch")
    from prototypes.predictor_adapters import make_mamba_predictor

    predictor = make_mamba_predictor()
    assert callable(predictor)


def test_make_hopfield_predictor_skips_without_torch() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("hflayers")
    from prototypes.predictor_adapters import make_hopfield_predictor

    predictor = make_hopfield_predictor(prefix_len=3)
    assert callable(predictor)


# ============================================================
# Report orchestration (synthetic data, no detector calls)
# ============================================================


def _make_synthetic_cutoff_metrics(
    cutoff: int | None,
    *,
    f1_by_detector: dict[str, float],
    trace_count: int = 100,
) -> CutoffMetrics:
    """Build a CutoffMetrics with TP/FP/FN/TN tuned to hit a target F1 per detector.

    For simplicity each detector gets ``tp = round(f1 * 50)``, ``fp = 50 - tp``,
    ``fn = 50 - tp``, ``tn = trace_count - 100 + tp``. This isn't a perfect F1
    reproduction but is monotonic and good enough for orchestration tests
    where we only care about ordering.
    """
    detector_metrics: dict[str, DetectorMetrics] = {}
    for detector in RUNNER_DETECTORS:
        f1 = f1_by_detector.get(detector, 0.5)
        tp = max(0, min(50, int(round(f1 * 50))))
        fp = 50 - tp
        fn = 50 - tp
        tn = max(0, trace_count - tp - fp - fn)
        detector_metrics[detector] = DetectorMetrics(detector=detector, tp=tp, fp=fp, fn=fn, tn=tn)
    return CutoffMetrics(
        cutoff=cutoff,
        trace_count=trace_count,
        skipped_short=0,
        skipped_min_steps=0,
        detector_metrics=detector_metrics,
        detectors_evaluated=tuple(RUNNER_DETECTORS),
    )


def test_cell_result_round_trip() -> None:
    from scripts.run_five_paradigm_comparative import CellResult

    cell = CellResult(
        paradigm="handcrafted",
        cutoff=3,
        detector="loop",
        tp=10,
        fp=2,
        fn=1,
        tn=87,
        precision=0.833,
        recall=0.909,
        f1=0.870,
        precision_lb=0.7,
        recall_lb=0.8,
        evaluated_traces=100,
    )
    payload = cell.as_dict()
    assert payload["paradigm"] == "handcrafted"
    assert payload["cutoff"] == 3
    assert payload["cutoff_label"] == "prefix-3"
    assert payload["detector"] == "loop"
    assert payload["f1"] == 0.870

    full_cell = replace(cell, cutoff=None)
    assert full_cell.as_dict()["cutoff_label"] == "full"
    assert full_cell.as_dict()["cutoff"] is None


def test_cells_from_cutoff_metrics_emits_one_cell_per_detector() -> None:
    from scripts.run_five_paradigm_comparative import _cells_from_cutoff_metrics

    metrics = _make_synthetic_cutoff_metrics(
        cutoff=5,
        f1_by_detector={"loop": 1.0, "stuck": 0.8, "confabulation": 0.6},
    )
    cells = _cells_from_cutoff_metrics("mamba", 5, metrics)
    assert len(cells) == len(RUNNER_DETECTORS)
    by_detector = {c.detector: c for c in cells}
    assert by_detector["loop"].paradigm == "mamba"
    assert by_detector["loop"].cutoff == 5
    # tp=50 fp=0 fn=0 → P=R=F1=1.0 for loop
    assert by_detector["loop"].tp == 50
    assert by_detector["loop"].f1 == pytest.approx(1.0)
    # tp=40 fp=10 fn=10 for stuck (f1≈0.8)
    assert by_detector["stuck"].tp == 40


def test_winner_for_picks_highest_f1() -> None:
    from scripts.run_five_paradigm_comparative import (
        CellResult,
        ComparativeReport,
        _winner_for,
    )

    cells = (
        CellResult("a", None, "loop", 1, 0, 0, 99, 1.0, 1.0, 1.0, 0.9, 0.9, 100),
        CellResult("b", None, "loop", 1, 1, 0, 98, 0.5, 1.0, 0.667, 0.5, 0.9, 100),
        CellResult("a", None, "stuck", 0, 0, 0, 100, 0.0, 0.0, 0.0, 0.0, 0.0, 100),
        CellResult("b", None, "stuck", 1, 0, 0, 99, 1.0, 1.0, 1.0, 0.9, 0.9, 100),
    )
    report = ComparativeReport(
        corpus_version="v1",
        min_confidence=0.8,
        paradigms=("a", "b"),
        cutoffs=(None,),
        detectors=("loop", "stuck"),
        filtered_trace_count=100,
        label_positive_counts={"loop": 1, "stuck": 1},
        cells=cells,
        timing_seconds={},
        generated_at="2026-04-07T00:00:00Z",
    )
    assert _winner_for(report, cutoff=None, detector="loop").paradigm == "a"
    assert _winner_for(report, cutoff=None, detector="stuck").paradigm == "b"


def test_render_markdown_smoke_produces_paper_sections() -> None:
    """End-to-end markdown render with a minimal synthetic report."""
    from scripts.run_five_paradigm_comparative import (
        CellResult,
        ComparativeReport,
        render_markdown,
    )

    paradigms = ("handcrafted", "causal", "tda", "mamba", "hopfield")
    detectors = ("loop", "stuck", "confabulation", "thrash", "runaway_cost")
    cutoffs: tuple[int | None, ...] = (3, None)
    cells: list[CellResult] = []
    for paradigm in paradigms:
        for cutoff in cutoffs:
            for detector in detectors:
                cells.append(
                    CellResult(
                        paradigm=paradigm,
                        cutoff=cutoff,
                        detector=detector,
                        tp=10,
                        fp=1,
                        fn=1,
                        tn=88,
                        precision=0.91,
                        recall=0.91,
                        f1=0.91,
                        precision_lb=0.85,
                        recall_lb=0.85,
                        evaluated_traces=100,
                    )
                )
    report = ComparativeReport(
        corpus_version="v1",
        min_confidence=0.8,
        paradigms=paradigms,
        cutoffs=cutoffs,
        detectors=detectors,
        filtered_trace_count=100,
        label_positive_counts={d: 11 for d in detectors},
        cells=tuple(cells),
        timing_seconds={f"{p}.total": 1.0 for p in paradigms},
        generated_at="2026-04-07T00:00:00Z",
    )
    md = render_markdown(report)

    # Spot-check the canonical paper-section headings.
    assert "# Five-Paradigm Comparative Report" in md
    assert "## TL;DR" in md
    assert "## Methodology" in md
    assert "## Full-Trace Comparison" in md
    assert "## Per-Detector Winners" in md
    assert "## Early-Detection F1 Curves" in md
    assert "## Per-Cutoff Winners" in md
    assert "## Macro-F1 Across Detectors" in md
    assert "## Failure Mode Analysis" in md
    assert "## Production Deployment Guidance" in md
    assert "## Reproducibility" in md
    # Spot-check that every paradigm appears in the rendered output.
    for paradigm in paradigms:
        assert paradigm in md
    for detector in detectors:
        assert detector in md
