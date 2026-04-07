"""Tests for the partial-trace evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import pytest
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from evaluator.partial_trace import (
    DETECTORS,
    CutoffMetrics,
    PartialTraceConfig,
    PartialTraceEvaluation,
    evaluate_at_cutoff,
    evaluate_partial_traces,
    format_partial_trace_table,
    make_handcrafted_predictor,
    truncate_trace,
)


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
# Config validation
# ============================================================


def test_config_default_cutoffs() -> None:
    config = PartialTraceConfig()
    assert config.cutoffs == (3, 5, 7, None)
    assert config.min_confidence == 0.8
    assert config.min_steps == 1
    assert config.skip_short_traces is False


def test_config_rejects_empty_cutoffs() -> None:
    with pytest.raises(ValueError):
        PartialTraceConfig(cutoffs=())


def test_config_rejects_zero_cutoff() -> None:
    with pytest.raises(ValueError):
        PartialTraceConfig(cutoffs=(0,))


def test_config_rejects_negative_cutoff() -> None:
    with pytest.raises(ValueError):
        PartialTraceConfig(cutoffs=(-1,))


def test_config_rejects_invalid_min_steps() -> None:
    with pytest.raises(ValueError):
        PartialTraceConfig(min_steps=0)


def test_config_rejects_invalid_min_confidence() -> None:
    with pytest.raises(ValueError):
        PartialTraceConfig(min_confidence=1.5)


def test_config_allows_full_only() -> None:
    config = PartialTraceConfig(cutoffs=(None,))
    assert config.cutoffs == (None,)


# ============================================================
# truncate_trace
# ============================================================


def test_truncate_returns_full_trace_for_none() -> None:
    trace = _make_trace(8)
    result = truncate_trace(trace, None)
    assert len(result) == 8
    assert result is not trace  # returns a copy


def test_truncate_to_smaller_than_length() -> None:
    trace = _make_trace(8)
    result = truncate_trace(trace, 3)
    assert len(result) == 3
    assert result[0].loop_index == 0
    assert result[2].loop_index == 2


def test_truncate_longer_than_trace_returns_full() -> None:
    trace = _make_trace(4)
    result = truncate_trace(trace, 10)
    assert len(result) == 4


def test_truncate_to_one_step() -> None:
    trace = _make_trace(5)
    result = truncate_trace(trace, 1)
    assert len(result) == 1
    assert result[0].loop_index == 0


def test_truncate_zero_raises() -> None:
    trace = _make_trace(5)
    with pytest.raises(ValueError):
        truncate_trace(trace, 0)


def test_truncate_negative_raises() -> None:
    trace = _make_trace(5)
    with pytest.raises(ValueError):
        truncate_trace(trace, -2)


# ============================================================
# evaluate_at_cutoff with synthetic predictor
# ============================================================


def _build_test_corpus(corpus_root: Path) -> None:
    """Write a tiny corpus directly as JSON files + manifest."""
    traces_dir = corpus_root / "traces" / "loop"
    pos_dir = traces_dir / "positive"
    neg_dir = traces_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Make 6 traces total: 3 positive, 3 negative; varying lengths
    manifest: list[dict[str, object]] = []
    for idx, length in enumerate([8, 6, 4]):  # positives
        snaps = [snap.model_dump(mode="json") for snap in _make_trace(length)]
        path = pos_dir / f"loop-pos-{idx:03d}.json"
        path.write_text(json.dumps(snaps))
        manifest.append(
            {
                "trace_id": f"loop-pos-{idx:03d}",
                "path": str(path.relative_to(corpus_root)),
                "tier": "synthetic",
                "labels": {
                    "loop": True,
                    "stuck": False,
                    "confabulation": False,
                    "thrash": False,
                    "runaway_cost": False,
                },
                "metadata": {"confidence": 1.0},
            }
        )
    for idx, length in enumerate([8, 6, 2]):  # negatives, last is too short
        snaps = [snap.model_dump(mode="json") for snap in _make_trace(length)]
        path = neg_dir / f"loop-neg-{idx:03d}.json"
        path.write_text(json.dumps(snaps))
        manifest.append(
            {
                "trace_id": f"loop-neg-{idx:03d}",
                "path": str(path.relative_to(corpus_root)),
                "tier": "synthetic",
                "labels": {
                    "loop": False,
                    "stuck": False,
                    "confabulation": False,
                    "thrash": False,
                    "runaway_cost": False,
                },
                "metadata": {"confidence": 1.0},
            }
        )

    (corpus_root / "manifest.json").write_text(json.dumps(manifest))


def _length_aware_predictor(snapshots: Sequence[VitalsSnapshot]) -> Mapping[str, bool]:
    """A predictor that fires 'loop' iff the trace has >= 5 steps.

    This lets us assert that truncation actually changes predictions:
    short prefixes should drop loop predictions, long prefixes should keep them.
    """
    fires = len(snapshots) >= 5
    return {detector: (fires if detector == "loop" else False) for detector in DETECTORS}


def test_evaluate_at_cutoff_with_custom_predictor(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        # At cutoff=3, the predictor should fire on no traces (len(prefix)=3 < 5)
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=3,
            predictor=_length_aware_predictor,
        )
    assert result.cutoff == 3
    assert result.trace_count == 6
    loop_metric = result.detector_metrics["loop"]
    # All positives missed: tp=0, fn=3, no false positives
    assert loop_metric.tp == 0
    assert loop_metric.fp == 0
    assert loop_metric.fn == 3
    assert loop_metric.tn == 3


def test_evaluate_at_cutoff_full_trace_fires_correctly(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=None,
            predictor=_length_aware_predictor,
        )
    loop_metric = result.detector_metrics["loop"]
    # Positives with length>=5: 8, 6 (yes), 4 (no). 2 TP, 1 FN
    assert loop_metric.tp == 2
    assert loop_metric.fn == 1
    # Negatives with length>=5: 8, 6 (yes — false positives), 2 (no — true negative)
    assert loop_metric.fp == 2
    assert loop_metric.tn == 1


def test_evaluate_at_cutoff_skip_short_traces(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=5,
            config=PartialTraceConfig(cutoffs=(5,), skip_short_traces=True),
            predictor=_length_aware_predictor,
        )
    # Traces with length < 5: pos[2] (len=4), neg[2] (len=2). Both skipped.
    assert result.trace_count == 4
    assert result.skipped_short == 2


def test_evaluate_at_cutoff_keeps_short_traces_by_default(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=5,
            predictor=_length_aware_predictor,
        )
    # All 6 traces participate; short ones use their full length
    assert result.trace_count == 6
    assert result.skipped_short == 0


def test_evaluate_at_cutoff_min_steps_filter(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=3,
            config=PartialTraceConfig(cutoffs=(3,), min_steps=5),
            predictor=_length_aware_predictor,
        )
    # min_steps=5 excludes pos[2] (len=4) and neg[2] (len=2)
    assert result.skipped_min_steps == 2
    assert result.trace_count == 4


# ============================================================
# evaluate_partial_traces (multi-cutoff)
# ============================================================


def test_evaluate_partial_traces_returns_one_entry_per_cutoff(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_partial_traces(
            corpus_version="v1",
            config=PartialTraceConfig(cutoffs=(3, 5, None)),
            predictor=_length_aware_predictor,
            predictor_label="length-aware",
        )
    assert isinstance(result, PartialTraceEvaluation)
    assert result.cutoffs == (3, 5, None)
    assert set(result.per_cutoff.keys()) == {3, 5, None}
    assert result.predictor_label == "length-aware"

    # Confirm metrics differ between cutoffs (3 misses everything, full catches some)
    assert result.metric(3, "loop").tp == 0
    assert result.metric(None, "loop").tp == 2


def test_evaluate_partial_traces_caches_traces_across_cutoffs(tmp_path: Path) -> None:
    """Each trace file should be loaded once even across multiple cutoffs."""
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    load_count = 0
    real_load = None
    from evaluator import partial_trace as pt_mod

    real_load = pt_mod.load_trace

    def counting_load(version: str, path: str) -> list[VitalsSnapshot]:
        nonlocal load_count
        load_count += 1
        return real_load(version, path)

    with (
        patch("evaluator.runner.CORPUS_ROOT", tmp_path),
        patch.object(pt_mod, "load_trace", counting_load),
    ):
        evaluate_partial_traces(
            corpus_version="v1",
            config=PartialTraceConfig(cutoffs=(3, 5, 7, None)),
            predictor=_length_aware_predictor,
        )

    # 6 traces loaded once, not 6 * 4 cutoffs = 24
    assert load_count == 6


def test_make_handcrafted_predictor_returns_dict() -> None:
    predictor = make_handcrafted_predictor()
    result = predictor(_make_trace(6))
    assert set(result.keys()) >= set(DETECTORS)
    for detector, value in result.items():
        assert isinstance(value, bool)


def test_make_handcrafted_predictor_handles_empty_trace() -> None:
    predictor = make_handcrafted_predictor()
    result = predictor([])
    for detector in DETECTORS:
        assert result[detector] is False


# ============================================================
# Confidence filtering
# ============================================================


def test_confidence_filter_excludes_low_confidence(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    corpus_root.mkdir(parents=True)

    snaps_high = [snap.model_dump(mode="json") for snap in _make_trace(6)]
    snaps_low = [snap.model_dump(mode="json") for snap in _make_trace(6)]
    (corpus_root / "high.json").write_text(json.dumps(snaps_high))
    (corpus_root / "low.json").write_text(json.dumps(snaps_low))

    manifest = [
        {
            "trace_id": "high",
            "path": "high.json",
            "labels": {d: False for d in DETECTORS},
            "metadata": {"confidence": 0.95},
        },
        {
            "trace_id": "low",
            "path": "low.json",
            "labels": {d: False for d in DETECTORS},
            "metadata": {"confidence": 0.5},
        },
    ]
    (corpus_root / "manifest.json").write_text(json.dumps(manifest))

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_at_cutoff(
            corpus_version="v1",
            cutoff=None,
            config=PartialTraceConfig(cutoffs=(None,), min_confidence=0.8),
            predictor=_length_aware_predictor,
        )
    assert result.trace_count == 1
    # Only the high-confidence trace participated. The length-aware predictor
    # fires loop=True on length>=5; label is False; so it counts as one FP.
    loop = result.detector_metrics["loop"]
    assert loop.tp + loop.fp + loop.fn + loop.tn == 1
    assert loop.fp == 1


def test_path_prefix_filter(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    confab_dir = corpus_root / "traces" / "confabulation" / "positive"
    loop_dir = corpus_root / "traces" / "loop" / "positive"
    confab_dir.mkdir(parents=True)
    loop_dir.mkdir(parents=True)

    snaps = [snap.model_dump(mode="json") for snap in _make_trace(6)]
    (confab_dir / "c.json").write_text(json.dumps(snaps))
    (loop_dir / "l.json").write_text(json.dumps(snaps))

    manifest = [
        {
            "trace_id": "c",
            "path": "traces/confabulation/positive/c.json",
            "labels": {d: False for d in DETECTORS},
            "metadata": {"confidence": 1.0},
        },
        {
            "trace_id": "l",
            "path": "traces/loop/positive/l.json",
            "labels": {d: False for d in DETECTORS},
            "metadata": {"confidence": 1.0},
        },
    ]
    (corpus_root / "manifest.json").write_text(json.dumps(manifest))

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_partial_traces(
            corpus_version="v1",
            config=PartialTraceConfig(
                cutoffs=(None,),
                path_prefixes=("traces/confabulation/",),
            ),
            predictor=_length_aware_predictor,
        )
    cm = result.per_cutoff[None]
    assert cm.trace_count == 1


# ============================================================
# CutoffMetrics + serialization
# ============================================================


def test_cutoff_metrics_label() -> None:
    full = CutoffMetrics(
        cutoff=None,
        trace_count=10,
        skipped_short=0,
        skipped_min_steps=0,
        detector_metrics={},
        detectors_evaluated=DETECTORS,
    )
    partial = CutoffMetrics(
        cutoff=5,
        trace_count=10,
        skipped_short=0,
        skipped_min_steps=0,
        detector_metrics={},
        detectors_evaluated=DETECTORS,
    )
    assert full.label == "full"
    assert partial.label == "prefix-5"


def test_partial_trace_evaluation_as_dict_round_trip(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_partial_traces(
            corpus_version="v1",
            config=PartialTraceConfig(cutoffs=(3, None)),
            predictor=_length_aware_predictor,
        )
    payload = result.as_dict()
    # Must be JSON-serializable
    serialized = json.dumps(payload)
    parsed = json.loads(serialized)
    assert parsed["corpus_version"] == "v1"
    assert len(parsed["per_cutoff"]) == 2
    assert parsed["per_cutoff"][0]["label"] == "prefix-3"
    assert parsed["per_cutoff"][1]["label"] == "full"


def test_format_partial_trace_table_renders(tmp_path: Path) -> None:
    corpus_root = tmp_path / "v1"
    _build_test_corpus(corpus_root)

    with patch("evaluator.runner.CORPUS_ROOT", tmp_path):
        result = evaluate_partial_traces(
            corpus_version="v1",
            config=PartialTraceConfig(cutoffs=(3, 5, None)),
            predictor=_length_aware_predictor,
        )
    table = format_partial_trace_table(result)
    lines = table.splitlines()
    # Header + separator + 5 detector rows
    assert len(lines) == 7
    assert "Detector" in lines[0]
    assert "3 steps" in lines[0]
    assert "Full" in lines[0]
    assert "loop" in table
