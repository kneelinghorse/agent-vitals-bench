"""Threshold-free TDA detector prototype for bench-side validation."""
# mypy: disable-error-code=import-not-found

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

DETECTORS: tuple[str, ...] = (
    "loop",
    "stuck",
    "confabulation",
    "thrash",
    "runaway_cost",
)
SIGNAL_KEYS: tuple[str, ...] = (
    "findings_count",
    "sources_count",
    "objectives_covered",
    "coverage_score",
    "total_tokens",
    "error_count",
    "confidence_score",
    "convergence_delta",
    "prompt_tokens",
    "completion_tokens",
    "refinement_count",
)
METRIC_KEYS: tuple[str, ...] = (
    "dm_coverage",
    "dm_findings",
    "cv_coverage",
    "cv_findings_rate",
    "qpf_tokens",
    "cs_effort",
)
DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "tda_models"


class MissingTDADependencyError(RuntimeError):
    """Raised when optional TDA dependencies are not installed."""


@dataclass(frozen=True)
class TDAConfig:
    corpus_root: Path = field(default_factory=lambda: DEFAULT_CORPUS_ROOT)
    window_sizes: tuple[int, ...] = (3, 4, 5)
    min_confidence: float = 0.8
    min_steps: int = 5
    homology_dimensions: tuple[int, ...] = (0, 1)
    n_splits: int = 5
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class TDAEvaluationMetric:
    detector: str
    f1: float
    precision: float
    recall: float
    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    artifact_path: str


@dataclass(frozen=True)
class TDATrainingSummary:
    trace_count: int
    skipped_traces: int
    feature_count: int
    metrics: dict[str, TDAEvaluationMetric]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "skipped_traces": self.skipped_traces,
            "feature_count": self.feature_count,
            "metrics": {detector: asdict(metric) for detector, metric in self.metrics.items()},
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True)
class TDAPrediction:
    detected: bool
    probability: float


@lru_cache(maxsize=1)
def _load_tda_backend() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        from gtda.diagrams import (
            Amplitude,
            BettiCurve,
            NumberOfPoints,
            PersistenceEntropy,
            PersistenceLandscape,
            Silhouette,
        )
        from gtda.homology import VietorisRipsPersistence
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import (
            StratifiedKFold,
            cross_val_predict,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise MissingTDADependencyError(
            "TDA dependencies are missing. Install `agent-vitals-bench[tda]` "
            "or use the sibling tda-experiment Python 3.12 environment."
        ) from exc

    return {
        "joblib": joblib,
        "np": np,
        "Amplitude": Amplitude,
        "BettiCurve": BettiCurve,
        "NumberOfPoints": NumberOfPoints,
        "PersistenceEntropy": PersistenceEntropy,
        "PersistenceLandscape": PersistenceLandscape,
        "Silhouette": Silhouette,
        "VietorisRipsPersistence": VietorisRipsPersistence,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "StratifiedKFold": StratifiedKFold,
        "cross_val_predict": cross_val_predict,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }


def _coerce_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _snapshot_section(snapshot: object, section: str) -> Mapping[str, Any]:
    if isinstance(snapshot, Mapping):
        return _coerce_mapping(snapshot.get(section, {}))
    return _coerce_mapping(getattr(snapshot, section, {}))


def _snapshot_row(snapshot: object) -> list[float]:
    signals = _snapshot_section(snapshot, "signals")
    metrics = _snapshot_section(snapshot, "metrics")
    row = [float(signals.get(key, 0.0) or 0.0) for key in SIGNAL_KEYS]
    row.extend(float(metrics.get(key, 0.0) or 0.0) for key in METRIC_KEYS)
    return row


def _trace_matrix(steps: Sequence[object], config: TDAConfig) -> Any | None:
    if len(steps) < config.min_steps:
        return None
    backend = _load_tda_backend()
    np = backend["np"]
    return np.array([_snapshot_row(step) for step in steps], dtype=np.float64)


def _normalize_trace(trace: Any, np: Any) -> Any:
    mins = trace.min(axis=0)
    ranges = trace.max(axis=0) - mins
    ranges[ranges == 0] = 1.0
    return (trace - mins) / ranges


def _build_point_cloud(trace_norm: Any, window_size: int, np: Any) -> Any:
    n_steps = int(trace_norm.shape[0])
    width = min(window_size, n_steps - 1)
    if width < 2:
        width = 2
    points = [trace_norm[idx : idx + width].flatten() for idx in range(n_steps - width + 1)]
    return np.array(points)


def _diagram_features(diagrams: Any, backend: Mapping[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features["entropy"] = backend["PersistenceEntropy"]().fit_transform(diagrams).flatten()
    features["amp_wasserstein"] = (
        backend["Amplitude"](
            metric="wasserstein",
            order=2,
        )
        .fit_transform(diagrams)
        .flatten()
    )
    features["amp_landscape"] = (
        backend["Amplitude"](metric="landscape").fit_transform(diagrams).flatten()
    )
    features["amp_bottleneck"] = (
        backend["Amplitude"](metric="bottleneck").fit_transform(diagrams).flatten()
    )
    features["n_points"] = backend["NumberOfPoints"]().fit_transform(diagrams).flatten()
    features["landscape"] = (
        backend["PersistenceLandscape"](
            n_layers=3,
            n_bins=20,
        )
        .fit_transform(diagrams)
        .flatten()
    )
    features["betti_curve"] = backend["BettiCurve"](n_bins=20).fit_transform(diagrams).flatten()
    features["silhouette"] = backend["Silhouette"](n_bins=20).fit_transform(diagrams).flatten()

    diag = diagrams[0]
    for dim in (0, 1):
        mask = diag[:, 2] == dim
        lifetimes: Any = backend["np"].array([], dtype=float)
        if mask.sum() > 0:
            lifetimes = diag[mask, 1] - diag[mask, 0]
            lifetimes = lifetimes[lifetimes > 0]
        prefix = f"h{dim}_lt"
        if len(lifetimes) > 0:
            features[f"{prefix}_mean"] = backend["np"].array([lifetimes.mean()])
            features[f"{prefix}_std"] = backend["np"].array([lifetimes.std()])
            features[f"{prefix}_max"] = backend["np"].array([lifetimes.max()])
            features[f"{prefix}_med"] = backend["np"].array([backend["np"].median(lifetimes)])
            features[f"{prefix}_skew"] = backend["np"].array(
                [
                    0.0
                    if lifetimes.std() == 0
                    else ((lifetimes - lifetimes.mean()) ** 3).mean() / (lifetimes.std() ** 3)
                ]
            )
            features[f"{prefix}_n_long"] = backend["np"].array(
                [float((lifetimes > backend["np"].percentile(lifetimes, 75)).sum())]
            )
        else:
            features[f"{prefix}_mean"] = backend["np"].array([0.0])
            features[f"{prefix}_std"] = backend["np"].array([0.0])
            features[f"{prefix}_max"] = backend["np"].array([0.0])
            features[f"{prefix}_med"] = backend["np"].array([0.0])
            features[f"{prefix}_skew"] = backend["np"].array([0.0])
            features[f"{prefix}_n_long"] = backend["np"].array([0.0])

    return features


def extract_tda_features(
    steps: Sequence[object],
    *,
    config: TDAConfig | None = None,
) -> list[float] | None:
    cfg = config or TDAConfig()
    matrix = _trace_matrix(steps, cfg)
    if matrix is None:
        return None

    backend = _load_tda_backend()
    np = backend["np"]
    trace_norm = _normalize_trace(matrix, np)

    all_features: dict[str, Any] = {}
    for window_size in cfg.window_sizes:
        point_cloud = _build_point_cloud(trace_norm, window_size, np)
        if len(point_cloud) < 3:
            return None
        vrp = backend["VietorisRipsPersistence"](
            homology_dimensions=cfg.homology_dimensions,
            max_edge_length=np.inf,
            n_jobs=1,
        )
        try:
            diagrams = vrp.fit_transform(point_cloud[np.newaxis, :, :])
        except Exception:
            return None
        diagram_features = _diagram_features(diagrams, backend)
        for name, values in diagram_features.items():
            all_features[f"w{window_size}_{name}"] = values

    vector = np.concatenate([all_features[name] for name in sorted(all_features.keys())])
    return [float(value) for value in vector]


def load_manifest_entries(*, config: TDAConfig | None = None) -> list[dict[str, Any]]:
    cfg = config or TDAConfig()
    manifest_path = cfg.corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [
        entry
        for entry in manifest
        if float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0) >= cfg.min_confidence
    ]


def _load_trace_steps(entry: Mapping[str, Any], config: TDAConfig) -> list[dict[str, Any]] | None:
    trace_path = config.corpus_root / str(entry["path"])
    if not trace_path.exists():
        return None
    steps = json.loads(trace_path.read_text())
    if not isinstance(steps, list):
        return None
    return steps


def build_feature_dataset(
    *,
    config: TDAConfig | None = None,
    manifest_entries: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, bool]], list[str], int, list[str]]:
    cfg = config or TDAConfig()
    backend = _load_tda_backend()
    np = backend["np"]
    entries = (
        list(manifest_entries)
        if manifest_entries is not None
        else load_manifest_entries(config=cfg)
    )

    features: list[list[float]] = []
    labels: list[dict[str, bool]] = []
    trace_ids: list[str] = []
    skipped = 0

    for entry in entries:
        steps = _load_trace_steps(entry, cfg)
        if steps is None:
            skipped += 1
            continue
        vector = extract_tda_features(steps, config=cfg)
        if vector is None:
            skipped += 1
            continue
        features.append(vector)
        labels.append({key: bool(entry["labels"].get(key, False)) for key in DETECTORS})
        trace_ids.append(str(entry["trace_id"]))

    if not features:
        raise ValueError("No traces produced usable TDA features")

    feature_names = [f"tda_feature_{idx}" for idx in range(len(features[0]))]
    return np.array(features), labels, trace_ids, skipped, feature_names


def train_tda_models(
    *,
    config: TDAConfig | None = None,
    detectors: Sequence[str] = DETECTORS,
    output_dir: Path = DEFAULT_MODEL_DIR,
) -> TDATrainingSummary:
    cfg = config or TDAConfig()
    backend = _load_tda_backend()
    np = backend["np"]
    X, labels, _trace_ids, skipped, feature_names = build_feature_dataset(config=cfg)

    metrics: dict[str, TDAEvaluationMetric] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for detector in detectors:
        y = np.array([1 if label.get(detector, False) else 0 for label in labels], dtype=int)
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        n_splits = min(cfg.n_splits, positives, negatives)
        if n_splits < 2:
            raise ValueError(f"detector {detector} does not have enough class balance for CV")

        pipeline = backend["Pipeline"](
            [
                ("scale", backend["StandardScaler"]()),
                (
                    "clf",
                    backend["GradientBoostingClassifier"](
                        n_estimators=cfg.n_estimators,
                        max_depth=cfg.max_depth,
                        learning_rate=cfg.learning_rate,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
        cv = backend["StratifiedKFold"](
            n_splits=n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )
        y_pred = backend["cross_val_predict"](pipeline, X, y, cv=cv)
        precision = float(backend["precision_score"](y, y_pred, zero_division=0))
        recall = float(backend["recall_score"](y, y_pred, zero_division=0))
        f1 = float(backend["f1_score"](y, y_pred, zero_division=0))
        tn, fp, fn, tp = backend["confusion_matrix"](y, y_pred, labels=[0, 1]).ravel()

        pipeline.fit(X, y)
        artifact_path = output_dir / f"{detector}.joblib"
        backend["joblib"].dump(
            {
                "detector": detector,
                "pipeline": pipeline,
                "feature_names": feature_names,
                "config": _config_to_dict(cfg),
            },
            artifact_path,
        )
        metrics[detector] = TDAEvaluationMetric(
            detector=detector,
            f1=f1,
            precision=precision,
            recall=recall,
            positives=positives,
            negatives=negatives,
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            artifact_path=str(artifact_path),
        )

    summary = TDATrainingSummary(
        trace_count=int(X.shape[0]),
        skipped_traces=skipped,
        feature_count=int(X.shape[1]),
        metrics=metrics,
        output_dir=str(output_dir),
    )
    (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    return summary


def load_tda_artifact(detector: str, *, model_dir: Path = DEFAULT_MODEL_DIR) -> Mapping[str, Any]:
    backend = _load_tda_backend()
    artifact_path = model_dir / f"{detector}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"TDA artifact not found: {artifact_path}")
    return cast(Mapping[str, Any], backend["joblib"].load(artifact_path))


def predict_detectors(
    steps: Sequence[object],
    *,
    detectors: Sequence[str] = DETECTORS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    config: TDAConfig | None = None,
) -> dict[str, TDAPrediction]:
    """Run all TDA detector models on a trace.

    Convenience wrapper that loads each detector's joblib artifact and runs
    a single prediction. For evaluation loops over many traces, prefer
    :func:`load_predictor` which loads each artifact exactly once and
    returns a fast per-trace callable.
    """
    return load_predictor(
        detectors=detectors,
        model_dir=model_dir,
        config=config,
    )(steps)


def load_predictor(
    *,
    detectors: Sequence[str] = DETECTORS,
    model_dir: Path = DEFAULT_MODEL_DIR,
    config: TDAConfig | None = None,
) -> Callable[[Sequence[object]], dict[str, TDAPrediction]]:
    """Load every detector's joblib pipeline once and return a cached predictor.

    The returned callable performs only the per-trace TDA feature extraction
    and ``predict_proba`` calls on each invocation; the joblib artifacts are
    loaded exactly once when this factory is called. Use this in evaluation
    loops over thousands of traces.

    The returned predictor still raises ``ValueError`` when feature
    extraction returns ``None`` (trace below the window-size floor), so
    callers can fall back to a different paradigm or skip the trace.
    """
    backend = _load_tda_backend()
    np = backend["np"]
    detector_list = tuple(detectors)
    pipelines = {
        detector: load_tda_artifact(detector, model_dir=model_dir)["pipeline"]
        for detector in detector_list
    }
    cfg = config

    def predict_one(steps: Sequence[object]) -> dict[str, TDAPrediction]:
        vector = extract_tda_features(steps, config=cfg)
        if vector is None:
            raise ValueError("trace did not have enough steps for TDA feature extraction")
        X = np.array([vector])
        result: dict[str, TDAPrediction] = {}
        for detector, pipeline in pipelines.items():
            proba = float(pipeline.predict_proba(X)[0][1])
            result[detector] = TDAPrediction(
                detected=bool(proba >= 0.5),
                probability=proba,
            )
        return result

    return predict_one


def _config_to_dict(config: TDAConfig) -> dict[str, Any]:
    data = asdict(config)
    data["corpus_root"] = str(config.corpus_root)
    return data


def main() -> None:
    summary = train_tda_models()
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
