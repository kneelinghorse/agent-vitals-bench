"""Tests for the TDA detector prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("joblib")
pytest.importorskip("gtda")
pytest.importorskip("sklearn")

from prototypes.tda_detector import (
    TDAConfig,
    extract_tda_features,
    predict_detectors,
    train_tda_models,
)


def _make_trace(kind: str, total_steps: int = 8) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for step in range(total_steps):
        if kind == "loop" and step >= 3:
            findings = 5
            sources = 2
            coverage = 0.46
            confidence = 0.84
        else:
            findings = step + 2
            sources = step + 5
            coverage = 0.35 + step * 0.08
            confidence = 0.3 + step * 0.07

        steps.append(
            {
                "signals": {
                    "findings_count": findings,
                    "sources_count": sources,
                    "objectives_covered": min(step + 1, 4),
                    "coverage_score": coverage,
                    "total_tokens": 400 * (step + 1),
                    "error_count": 0,
                    "confidence_score": confidence,
                    "convergence_delta": 0.12 if kind == "healthy" else 0.03,
                    "prompt_tokens": int(400 * (step + 1) * 0.6),
                    "completion_tokens": int(400 * (step + 1) * 0.4),
                    "refinement_count": 0,
                },
                "metrics": {
                    "dm_coverage": 0.68 if kind == "healthy" else 0.24,
                    "dm_findings": 0.62 if kind == "healthy" else 0.08,
                    "cv_coverage": 0.1,
                    "cv_findings_rate": 0.12 if kind == "healthy" else 0.02,
                    "qpf_tokens": 0.58,
                    "cs_effort": 0.52 if kind == "healthy" else 0.3,
                },
            }
        )
    return steps


def _write_training_corpus(corpus_root: Path) -> None:
    positive_dir = corpus_root / "traces" / "loop" / "positive"
    negative_dir = corpus_root / "traces" / "loop" / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for idx in range(4):
        trace_id = f"loop-pos-{idx:03d}"
        path = positive_dir / f"{trace_id}.json"
        path.write_text(json.dumps(_make_trace("loop"), indent=2) + "\n")
        manifest.append(
            {
                "trace_id": trace_id,
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
    for idx in range(4):
        trace_id = f"loop-neg-{idx:03d}"
        path = negative_dir / f"{trace_id}.json"
        path.write_text(json.dumps(_make_trace("healthy"), indent=2) + "\n")
        manifest.append(
            {
                "trace_id": trace_id,
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

    (corpus_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test_extract_tda_features_returns_vector_for_valid_trace() -> None:
    features = extract_tda_features(_make_trace("healthy"))

    assert features is not None
    assert len(features) > 0


def test_extract_tda_features_returns_none_for_short_trace() -> None:
    assert extract_tda_features(_make_trace("healthy", total_steps=4)) is None


def test_train_tda_models_writes_artifact_and_predicts(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus" / "v1"
    _write_training_corpus(corpus_root)
    output_dir = tmp_path / "models"

    summary = train_tda_models(
        config=TDAConfig(
            corpus_root=corpus_root,
            n_splits=2,
            n_estimators=20,
            max_depth=2,
        ),
        detectors=("loop",),
        output_dir=output_dir,
    )

    assert summary.trace_count == 8
    assert "loop" in summary.metrics
    assert (output_dir / "loop.joblib").exists()
    assert (output_dir / "summary.json").exists()

    prediction = predict_detectors(
        _make_trace("loop"),
        detectors=("loop",),
        model_dir=output_dir,
        config=TDAConfig(corpus_root=corpus_root),
    )["loop"]
    assert 0.0 <= prediction.probability <= 1.0
