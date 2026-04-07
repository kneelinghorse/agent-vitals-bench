"""Tests for the Hopfield detector prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototypes.hopfield_detector import (
    DEFAULT_PREFIX_CUTOFFS,
    DETECTORS,
    METRIC_KEYS,
    N_FEATURES,
    SIGNAL_KEYS,
    HopfieldConfig,
    HopfieldPrediction,
    _snapshot_row,
    load_manifest_entries,
)


def _make_trace(kind: str, total_steps: int = 8) -> list[dict[str, object]]:
    """Synthetic trace shaped like a loop or healthy agent."""
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
    for idx in range(10):
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
    for idx in range(10):
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


# ============================================================
# Pure-stdlib tests (always run)
# ============================================================


def test_constants_are_defined() -> None:
    assert len(DETECTORS) == 5
    assert N_FEATURES == len(SIGNAL_KEYS) + len(METRIC_KEYS)
    assert DEFAULT_PREFIX_CUTOFFS == (3, 5, 7)


def test_snapshot_row_extracts_features_in_order() -> None:
    snapshot = {
        "signals": {key: float(idx + 1) for idx, key in enumerate(SIGNAL_KEYS)},
        "metrics": {key: float(idx + 100) for idx, key in enumerate(METRIC_KEYS)},
    }
    row = _snapshot_row(snapshot)
    assert len(row) == N_FEATURES
    for idx in range(len(SIGNAL_KEYS)):
        assert row[idx] == float(idx + 1)
    for idx in range(len(METRIC_KEYS)):
        assert row[len(SIGNAL_KEYS) + idx] == float(idx + 100)


def test_snapshot_row_handles_missing_and_none() -> None:
    row_empty = _snapshot_row({"signals": {}, "metrics": {}})
    assert all(v == 0.0 for v in row_empty)

    row_none = _snapshot_row(
        {
            "signals": {key: None for key in SIGNAL_KEYS},
            "metrics": {key: None for key in METRIC_KEYS},
        }
    )
    assert all(v == 0.0 for v in row_none)


def test_load_manifest_entries_filters_by_confidence(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus" / "v1"
    corpus_root.mkdir(parents=True)
    manifest = [
        {
            "trace_id": "high",
            "path": "traces/x.json",
            "labels": {},
            "metadata": {"confidence": 0.9},
        },
        {
            "trace_id": "low",
            "path": "traces/y.json",
            "labels": {},
            "metadata": {"confidence": 0.5},
        },
    ]
    (corpus_root / "manifest.json").write_text(json.dumps(manifest))

    entries = load_manifest_entries(config=HopfieldConfig(corpus_root=corpus_root))
    trace_ids = {entry["trace_id"] for entry in entries}
    assert trace_ids == {"high"}


def test_hopfield_config_defaults() -> None:
    config = HopfieldConfig()
    assert config.d_model == 64
    assert config.n_stored == 32
    assert config.n_heads == 4
    assert config.scaling == 0.25
    assert config.epochs == 30
    assert config.n_splits == 5
    assert config.min_confidence == 0.8
    assert config.min_steps == 5
    assert config.max_steps == 20


def test_hopfield_prediction_dataclass() -> None:
    pred = HopfieldPrediction(detected=True, probability=0.91, energy=-2.5)
    assert pred.detected is True
    assert pred.probability == 0.91
    assert pred.energy == -2.5
    # energy is optional
    pred2 = HopfieldPrediction(detected=False, probability=0.1)
    assert pred2.energy is None


# ============================================================
# Torch+hflayers-gated tests (each marked individually)
# ============================================================


def _have_hopfield_deps() -> bool:
    try:
        import hflayers  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


requires_hopfield = pytest.mark.skipif(
    not _have_hopfield_deps(),
    reason="requires torch + hflayers + sklearn + numpy (install with `[hopfield]` extra)",
)


@requires_hopfield
def test_classifier_param_count_matches_research_target() -> None:
    """The Hopfield classifier with d_model=64, n_stored=32 should be ~137K params."""
    from prototypes.hopfield_detector import _build_classifier_class

    Classifier = _build_classifier_class()
    model = Classifier(n_features=N_FEATURES, d_model=64, n_stored=32, n_heads=4)
    params = model.param_count()
    # Research target is 136,897 params
    assert 130_000 <= params <= 145_000, f"Expected ~137K params, got {params}"


@requires_hopfield
def test_classifier_forward_shape() -> None:
    import torch

    from prototypes.hopfield_detector import _build_classifier_class

    Classifier = _build_classifier_class()
    model = Classifier(n_features=N_FEATURES, d_model=32, n_stored=16, n_heads=2)
    x = torch.randn(4, 8, N_FEATURES)
    lengths = torch.tensor([8, 6, 5, 7], dtype=torch.long)
    out = model(x, lengths)
    assert out.shape == (4,)


@requires_hopfield
def test_classifier_energy_shape() -> None:
    import torch

    from prototypes.hopfield_detector import _build_classifier_class

    Classifier = _build_classifier_class()
    model = Classifier(n_features=N_FEATURES, d_model=32, n_stored=16, n_heads=2)
    x = torch.randn(3, 8, N_FEATURES)
    lengths = torch.tensor([8, 7, 5], dtype=torch.long)
    energies = model.get_energy(x, lengths)
    assert energies.shape == (3,)


@requires_hopfield
def test_build_trace_dataset_loads_from_corpus(tmp_path: Path) -> None:
    from prototypes.hopfield_detector import build_trace_dataset

    corpus_root = tmp_path / "corpus" / "v1"
    _write_training_corpus(corpus_root)

    traces, labels, trace_ids, skipped = build_trace_dataset(
        config=HopfieldConfig(corpus_root=corpus_root)
    )
    assert len(traces) == 20
    assert skipped == 0
    assert traces[0].shape[1] == N_FEATURES
    assert sum(label["loop"] for label in labels) == 10


@requires_hopfield
def test_train_hopfield_full_writes_artifact_and_predicts(tmp_path: Path) -> None:
    from prototypes.hopfield_detector import predict, train_hopfield_models

    corpus_root = tmp_path / "corpus" / "v1"
    _write_training_corpus(corpus_root)
    output_dir = tmp_path / "models"

    summary = train_hopfield_models(
        config=HopfieldConfig(
            corpus_root=corpus_root,
            n_splits=2,
            epochs=3,
            d_model=16,
            n_stored=8,
            n_heads=2,
            device="cpu",
        ),
        detectors=("loop",),
        prefix_len=None,
        output_dir=output_dir,
    )

    assert summary.trace_count == 20
    assert summary.mode == "full"
    assert "loop" in summary.metrics
    assert (output_dir / "loop.pt").exists()
    assert (output_dir / "summary_full.json").exists()

    prediction = predict(_make_trace("loop"), "loop", model_dir=output_dir)
    assert isinstance(prediction, HopfieldPrediction)
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.energy is not None


@requires_hopfield
def test_train_hopfield_prefix_writes_artifact_and_predicts(tmp_path: Path) -> None:
    from prototypes.hopfield_detector import predict, train_hopfield_models

    corpus_root = tmp_path / "corpus" / "v1"
    _write_training_corpus(corpus_root)
    output_dir = tmp_path / "models"

    summary = train_hopfield_models(
        config=HopfieldConfig(
            corpus_root=corpus_root,
            n_splits=2,
            epochs=3,
            d_model=16,
            n_stored=8,
            n_heads=2,
            device="cpu",
        ),
        detectors=("loop",),
        prefix_len=5,
        output_dir=output_dir,
    )

    assert summary.mode == "prefix-5"
    assert "loop" in summary.metrics
    assert summary.metrics["loop"].prefix_len == 5
    assert (output_dir / "loop_p5.pt").exists()
    assert (output_dir / "summary_prefix-5.json").exists()

    prediction = predict(
        _make_trace("loop"),
        "loop",
        prefix_len=5,
        model_dir=output_dir,
    )
    assert 0.0 <= prediction.probability <= 1.0


@requires_hopfield
def test_load_artifact_raises_on_missing(tmp_path: Path) -> None:
    from prototypes.hopfield_detector import load_hopfield_artifact

    with pytest.raises(FileNotFoundError):
        load_hopfield_artifact("nonexistent", model_dir=tmp_path)


@requires_hopfield
def test_artifact_round_trip_preserves_config(tmp_path: Path) -> None:
    from prototypes.hopfield_detector import load_hopfield_artifact, train_hopfield_models

    corpus_root = tmp_path / "corpus" / "v1"
    _write_training_corpus(corpus_root)
    output_dir = tmp_path / "models"

    cfg = HopfieldConfig(
        corpus_root=corpus_root,
        n_splits=2,
        epochs=2,
        d_model=16,
        n_stored=8,
        n_heads=2,
        device="cpu",
    )
    train_hopfield_models(
        config=cfg,
        detectors=("loop",),
        prefix_len=3,
        output_dir=output_dir,
    )

    artifact = load_hopfield_artifact("loop", prefix_len=3, model_dir=output_dir)
    assert artifact["mode"] == "prefix-3"
    assert artifact["prefix_len"] == 3
    assert artifact["detector"] == "loop"
    assert artifact["n_features"] == N_FEATURES
    assert artifact["config"]["d_model"] == 16
    assert artifact["config"]["n_stored"] == 8
