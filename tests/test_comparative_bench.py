"""Tests for the comparative benchmark harness."""

from __future__ import annotations

from pathlib import Path

from prototypes.comparative_bench import (
    AgreementSummary,
    ApproachSummary,
    ComparativeBenchmarkResult,
    DetectorComparison,
    FailureAnalysis,
    FoldMetricSummary,
    MetricSummary,
    Recommendation,
    TraceRecord,
    _load_raw_steps,
    build_agreement_summary,
    build_fold_metrics,
    render_markdown_report,
    summarize_failures,
)


def _labels(**overrides: bool) -> dict[str, bool]:
    labels = {
        "loop": False,
        "stuck": False,
        "confabulation": False,
        "thrash": False,
        "runaway_cost": False,
    }
    labels.update(overrides)
    return labels


def _predictions(
    *,
    handcrafted: dict[str, bool | None] | None = None,
    causal: dict[str, bool | None] | None = None,
    tda: dict[str, bool | None] | None = None,
) -> dict[str, dict[str, bool | None]]:
    return {
        "handcrafted": handcrafted or {},
        "causal": causal or {},
        "tda": tda or {},
    }


def _details(
    *,
    causal_confab: dict[str, float | str | None] | None = None,
    causal_runaway: dict[str, float | str | None] | None = None,
) -> dict[str, dict[str, dict[str, float | str | None]]]:
    return {
        "causal": {
            "confabulation": causal_confab or {},
            "runaway_cost": causal_runaway or {},
        },
        "tda": {},
    }


def _record(
    trace_id: str,
    *,
    path: str,
    labels: dict[str, bool],
    predictions: dict[str, dict[str, bool | None]],
    delayed_onset: bool = False,
    details: dict[str, dict[str, dict[str, float | str | None]]] | None = None,
) -> TraceRecord:
    return TraceRecord(
        trace_id=trace_id,
        path=path,
        tier="synthetic",
        confidence=1.0,
        step_count=8,
        family="synthetic",
        delayed_onset=delayed_onset,
        labels=labels,
        predictions=predictions,
        details=details or _details(),
    )


def test_load_raw_steps_supports_json_array_and_jsonl(tmp_path: Path) -> None:
    array_path = tmp_path / "array.json"
    array_path.write_text('[{"signals": {"findings_count": 1}}]\n')
    jsonl_path = tmp_path / "trace.jsonl"
    jsonl_path.write_text(
        '{"signals": {"findings_count": 1}}\n{"signals": {"findings_count": 2}}\n'
    )

    array_steps = _load_raw_steps(array_path)
    jsonl_steps = _load_raw_steps(jsonl_path)

    assert array_steps is not None
    assert jsonl_steps is not None
    assert len(array_steps) == 1
    assert len(jsonl_steps) == 2


def test_build_fold_metrics_reports_perfect_scores_for_balanced_records() -> None:
    records = [
        _record(
            f"loop-pos-{index}",
            path=f"traces/loop/positive/loop-pos-{index}.json",
            labels=_labels(loop=True),
            predictions=_predictions(handcrafted={"loop": True}),
        )
        for index in range(5)
    ] + [
        _record(
            f"loop-neg-{index}",
            path=f"traces/loop/negative/loop-neg-{index}.json",
            labels=_labels(loop=False),
            predictions=_predictions(handcrafted={"loop": False}),
        )
        for index in range(5)
    ]

    fold_metrics = build_fold_metrics(
        records,
        detector="loop",
        approach="handcrafted",
        n_splits=5,
        seed=42,
    )

    assert len(fold_metrics) == 5
    assert all(fold.trace_count == 2 for fold in fold_metrics)
    assert all(fold.metric.f1 == 1.0 for fold in fold_metrics)


def test_summarize_failures_uses_causal_reason_buckets() -> None:
    record = _record(
        "confab-delayed-001",
        path="traces/confabulation/positive/confab-syn-delayed-001.json",
        labels=_labels(confabulation=True),
        predictions=_predictions(causal={"confabulation": False}),
        delayed_onset=True,
        details=_details(
            causal_confab={
                "baseline_strength": 0.65,
                "weakest_strength": 0.5,
                "structural_drop": 0.15,
                "final_ratio": 0.1,
                "trigger": None,
            }
        ),
    )

    analysis = summarize_failures([record], detector="confabulation", approach="causal")

    assert analysis is not None
    assert analysis.false_negatives == 1
    assert analysis.fn_buckets[0].bucket == "link_never_weakened+delayed_onset"


def test_build_agreement_summary_counts_signatures_and_pairwise_rates() -> None:
    records = [
        _record(
            "all-hit",
            path="traces/confabulation/positive/confab-1.json",
            labels=_labels(confabulation=True),
            predictions=_predictions(
                handcrafted={"confabulation": True},
                causal={"confabulation": True},
                tda={"confabulation": True},
            ),
        ),
        _record(
            "tda-only",
            path="traces/confabulation/positive/confab-2.json",
            labels=_labels(confabulation=True),
            predictions=_predictions(
                handcrafted={"confabulation": False},
                causal={"confabulation": False},
                tda={"confabulation": True},
            ),
        ),
        _record(
            "neg-clean",
            path="traces/confabulation/negative/confab-3.json",
            labels=_labels(confabulation=False),
            predictions=_predictions(
                handcrafted={"confabulation": False},
                causal={"confabulation": False},
                tda={"confabulation": False},
            ),
        ),
    ]

    summary = build_agreement_summary(
        records,
        detector="confabulation",
        approaches=("handcrafted", "causal", "tda"),
    )

    assert summary.aligned_trace_count == 3
    assert summary.positive_signatures["handcrafted+causal+tda"] == 1
    assert summary.positive_signatures["tda"] == 1
    assert summary.negative_signatures["none"] == 1
    assert any(pair.left == "handcrafted" and pair.right == "causal" for pair in summary.pairwise)
    assert len(summary.disagreement_examples) == 1


def test_render_markdown_report_includes_comparison_and_recommendation_sections() -> None:
    metric = MetricSummary.from_counts(tp=9, fp=1, fn=1, tn=9, evaluated_traces=20)
    failure = FailureAnalysis(false_negatives=1, false_positives=1, fn_buckets=(), fp_buckets=())
    detector = DetectorComparison(
        detector="confabulation",
        approach_summaries=(
            ApproachSummary(
                approach="handcrafted",
                implemented=True,
                aligned_metric=metric,
                full_metric=metric,
                fold_metrics=(FoldMetricSummary(fold_index=1, trace_count=4, metric=metric),),
                full_trace_count=20,
                aligned_trace_count=20,
                coverage_rate=1.0,
                failure_analysis=failure,
            ),
            ApproachSummary(
                approach="tda",
                implemented=True,
                aligned_metric=metric,
                full_metric=metric,
                fold_metrics=(),
                full_trace_count=18,
                aligned_trace_count=18,
                coverage_rate=0.9,
                failure_analysis=failure,
            ),
        ),
        agreement=AgreementSummary(
            aligned_trace_count=18,
            available_approaches=("handcrafted", "tda"),
            pairwise=(),
            positive_signatures={"handcrafted+tda": 7},
            negative_signatures={"none": 10},
            disagreement_examples=(),
        ),
        recommendation=Recommendation(
            recommended_approach="tda",
            rationale="TDA wins the aligned comparison.",
            hybrid_option="Keep causal as an explainer.",
        ),
    )
    result = ComparativeBenchmarkResult(
        corpus_version="v1",
        min_confidence=0.8,
        generated_at="2026-04-06 19:00 UTC",
        filtered_trace_count=20,
        tda_trace_count=18,
        tda_excluded_count=2,
        tda_excluded_reasons={"insufficient_steps": 2},
        detectors=(detector,),
        trace_records=(),
    )

    markdown = render_markdown_report(result)

    assert "# Causal vs TDA Comparative Benchmark" in markdown
    assert "## Per-Detector Comparison" in markdown
    assert "| confabulation | handcrafted | 20/20 |" in markdown
    assert "| confabulation | tda | 18/20 |" in markdown
    assert "## Recommendations" in markdown
    assert "TDA wins the aligned comparison." in markdown
