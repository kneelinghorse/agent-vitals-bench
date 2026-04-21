"""Modern Hopfield network behavioral classifier prototype.

Uses learnable stored patterns (behavioral archetypes) plus Hopfield
self-attention for trace classification. The key capability is **early
detection** from partial traces — the model can predict failure modes after
seeing only the first 3, 5, or 7 steps of a trace.

Architecture: input projection (n_features -> d_model) + 32 learnable stored
patterns + 2x Hopfield self-attention layers (4 heads each) + classification
head. ~137K params with d_model=64. Energy landscape exposes anomaly score:
low energy = recognized pattern, high energy = novel/anomalous.

Dependencies are gated: torch, hflayers (hopfield-layers), scikit-learn, and
numpy are loaded lazily via ``_load_hopfield_backend()``. Install via
``pip install -e .[hopfield]`` or use the sibling ``tda-experiment`` Python
3.12 venv which already has the stack.
"""

# mypy: disable-error-code=import-not-found

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
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
N_FEATURES: int = len(SIGNAL_KEYS) + len(METRIC_KEYS)
DEFAULT_PREFIX_CUTOFFS: tuple[int, ...] = (3, 5, 7)
DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "hopfield_models"


class MissingHopfieldDependencyError(RuntimeError):
    """Raised when optional Hopfield dependencies are missing."""


@dataclass(frozen=True)
class HopfieldConfig:
    """Tunable parameters for the Hopfield detector prototype."""

    corpus_root: Path = field(default_factory=lambda: DEFAULT_CORPUS_ROOT)
    min_confidence: float = 0.8
    min_steps: int = 5
    max_steps: int = 20
    d_model: int = 64
    n_stored: int = 32
    n_heads: int = 4
    scaling: float = 0.25
    dropout: float = 0.1
    n_splits: int = 5
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    random_state: int = 42
    device: str | None = None


@dataclass(frozen=True)
class HopfieldEvaluationMetric:
    """Per-detector evaluation metrics, optionally tagged with a step cutoff."""

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
    f1_std: float = 0.0
    prefix_len: int | None = None


@dataclass(frozen=True)
class HopfieldTrainingSummary:
    """Aggregate training summary for one Hopfield run."""

    trace_count: int
    skipped_traces: int
    feature_count: int
    model_params: int
    mode: str  # "full" or f"prefix-{N}"
    metrics: dict[str, HopfieldEvaluationMetric]
    output_dir: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "skipped_traces": self.skipped_traces,
            "feature_count": self.feature_count,
            "model_params": self.model_params,
            "mode": self.mode,
            "metrics": {detector: asdict(metric) for detector, metric in self.metrics.items()},
            "output_dir": self.output_dir,
            "device": self.device,
        }


@dataclass(frozen=True)
class HopfieldPrediction:
    """Single prediction output."""

    detected: bool
    probability: float
    energy: float | None = None


@lru_cache(maxsize=1)
def _load_hopfield_backend() -> dict[str, Any]:
    """Lazy import of torch, hflayers, sklearn, and numpy."""
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from hflayers import Hopfield
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise MissingHopfieldDependencyError(
            "Hopfield dependencies are missing. Install `agent-vitals-bench[hopfield]` "
            "or use the sibling tda-experiment Python 3.12 environment."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "Hopfield": Hopfield,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "StratifiedKFold": StratifiedKFold,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "confusion_matrix": confusion_matrix,
    }


def _resolve_device(device_pref: str | None) -> Any:
    backend = _load_hopfield_backend()
    torch = backend["torch"]
    if device_pref is not None:
        return torch.device(device_pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Model architecture (built lazily inside factory functions)
# ============================================================


def _build_classifier_class() -> Any:
    """Build the HopfieldClassifier class lazily once torch is available."""
    backend = _load_hopfield_backend()
    nn = backend["nn"]
    torch = backend["torch"]
    Hopfield = backend["Hopfield"]

    class HopfieldClassifier(nn.Module):  # type: ignore[misc, name-defined]
        """Modern Hopfield classifier with learnable behavioral archetypes."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_stored: int = 32,
            n_heads: int = 4,
            scaling: float = 0.25,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_stored = n_stored

            self.input_proj = nn.Linear(n_features, d_model)
            self.stored_patterns = nn.Parameter(torch.randn(1, n_stored, d_model) * 0.02)

            self.hopfield1 = Hopfield(
                input_size=d_model,
                hidden_size=d_model,
                num_heads=n_heads,
                scaling=scaling,
                dropout=dropout,
            )
            self.hopfield2 = Hopfield(
                input_size=d_model,
                hidden_size=d_model,
                num_heads=n_heads,
                scaling=scaling,
                dropout=dropout,
            )

            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )

        def forward(self, x: Any, lengths: Any | None = None) -> Any:
            batch_size, _seq_len, _ = x.shape
            x = self.input_proj(x)

            stored = self.stored_patterns.expand(batch_size, -1, -1)
            combined = torch.cat([stored, x], dim=1)
            combined = self.hopfield1(combined)
            combined = self.hopfield2(combined)

            x_out = combined[:, self.n_stored :, :]

            if lengths is not None:
                idx = (lengths - 1).long().clamp(min=0)
                idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_model)
                pooled = x_out.gather(1, idx).squeeze(1)
            else:
                pooled = x_out[:, -1, :]

            return self.head(pooled).squeeze(-1)

        def get_energy(self, x: Any, lengths: Any | None = None) -> Any:
            """Hopfield energy: lower = better match against stored patterns."""
            batch_size = x.shape[0]
            x_proj = self.input_proj(x)
            stored = self.stored_patterns.expand(batch_size, -1, -1)
            sim = torch.bmm(x_proj, stored.transpose(1, 2)) * 0.25
            energies = -torch.logsumexp(sim, dim=-1)
            if lengths is not None:
                mask = torch.arange(energies.shape[1], device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                energies = (energies * mask.float()).sum(dim=1) / lengths.float()
            else:
                energies = energies.mean(dim=1)
            return energies

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return HopfieldClassifier


def _build_dataset_class() -> Any:
    """Build the TraceDataset class with optional prefix truncation."""
    backend = _load_hopfield_backend()
    Dataset = backend["Dataset"]
    torch = backend["torch"]
    np = backend["np"]

    class TraceDataset(Dataset):  # type: ignore[misc, valid-type]
        """Pads/truncates traces, optionally to a partial prefix length."""

        def __init__(
            self,
            traces: list[Any],
            labels: Any,
            max_len: int,
            prefix_len: int | None = None,
        ) -> None:
            self.traces = traces
            self.labels = labels
            self.max_len = max_len
            self.prefix_len = prefix_len

        def __len__(self) -> int:
            return len(self.traces)

        def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
            trace = self.traces[idx]
            label = self.labels[idx]
            actual_len = len(trace)
            if self.prefix_len is not None:
                use_len = min(self.prefix_len, actual_len)
            else:
                use_len = min(actual_len, self.max_len)

            padded = np.zeros((self.max_len, trace.shape[1]), dtype=np.float32)
            padded[:use_len] = trace[:use_len]

            return (
                torch.tensor(padded, dtype=torch.float32),
                torch.tensor(use_len, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    return TraceDataset


# ============================================================
# Data loading
# ============================================================


def _coerce_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _snapshot_row(snapshot: Mapping[str, Any]) -> list[float]:
    """Extract a feature row from a single snapshot dict."""
    signals = _coerce_mapping(snapshot.get("signals", {}))
    metrics = _coerce_mapping(snapshot.get("metrics", {}))
    row = [float(signals.get(key, 0.0) or 0.0) for key in SIGNAL_KEYS]
    row.extend(float(metrics.get(key, 0.0) or 0.0) for key in METRIC_KEYS)
    return row


def load_manifest_entries(*, config: HopfieldConfig | None = None) -> list[dict[str, Any]]:
    """Load manifest entries that meet the minimum confidence threshold."""
    cfg = config or HopfieldConfig()
    manifest_path = cfg.corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [
        entry
        for entry in manifest
        if float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0) >= cfg.min_confidence
    ]


def _load_trace_steps(
    entry: Mapping[str, Any], config: HopfieldConfig
) -> list[dict[str, Any]] | None:
    trace_path = config.corpus_root / str(entry["path"])
    if not trace_path.exists():
        return None
    steps = json.loads(trace_path.read_text())
    if not isinstance(steps, list):
        return None
    return steps


def build_trace_dataset(
    *,
    config: HopfieldConfig | None = None,
    manifest_entries: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Any], list[dict[str, bool]], list[str], int]:
    """Load traces from the corpus into per-trace numpy matrices."""
    cfg = config or HopfieldConfig()
    backend = _load_hopfield_backend()
    np = backend["np"]
    entries = (
        list(manifest_entries)
        if manifest_entries is not None
        else load_manifest_entries(config=cfg)
    )

    traces: list[Any] = []
    labels: list[dict[str, bool]] = []
    trace_ids: list[str] = []
    skipped = 0

    for entry in entries:
        steps = _load_trace_steps(entry, cfg)
        if steps is None or len(steps) < cfg.min_steps:
            skipped += 1
            continue
        rows = [_snapshot_row(step) for step in steps]
        traces.append(np.array(rows, dtype=np.float32))
        labels.append({key: bool(entry["labels"].get(key, False)) for key in DETECTORS})
        trace_ids.append(str(entry["trace_id"]))

    if not traces:
        raise ValueError("No traces loaded from corpus")

    return traces, labels, trace_ids, skipped


def _compute_normalization(traces: Sequence[Any]) -> tuple[Any, Any]:
    backend = _load_hopfield_backend()
    np = backend["np"]
    all_steps = np.vstack(list(traces))
    return all_steps.mean(axis=0), all_steps.std(axis=0) + 1e-8


def _normalize(traces: Sequence[Any], mean: Any, std: Any) -> list[Any]:
    return [(t - mean) / std for t in traces]


# ============================================================
# Training
# ============================================================


def _train_one_fold(
    *,
    train_traces: list[Any],
    val_traces: list[Any],
    y_train: Any,
    y_val: Any,
    config: HopfieldConfig,
    prefix_len: int | None,
    device: Any,
) -> Any:
    """Train a single fold and return best validation predictions."""
    backend = _load_hopfield_backend()
    F = backend["F"]
    torch = backend["torch"]
    DataLoader = backend["DataLoader"]
    f1_score = backend["f1_score"]
    np = backend["np"]

    mean, std = _compute_normalization(train_traces)
    train_norm = _normalize(train_traces, mean, std)
    val_norm = _normalize(val_traces, mean, std)

    TraceDataset = _build_dataset_class()
    train_ds = TraceDataset(train_norm, y_train, config.max_steps, prefix_len=prefix_len)
    val_ds = TraceDataset(val_norm, y_val, config.max_steps, prefix_len=prefix_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    Classifier = _build_classifier_class()
    model = Classifier(
        n_features=N_FEATURES,
        d_model=config.d_model,
        n_stored=config.n_stored,
        n_heads=config.n_heads,
        scaling=config.scaling,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_f1 = -1.0
    best_preds: Any = None

    for _epoch in range(config.epochs):
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds: list[Any] = []
        with torch.no_grad():
            for x, lengths, _y in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                logits = model(x, lengths)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                all_preds.append(preds)
        fold_preds = np.concatenate(all_preds, axis=0)
        current_f1 = float(f1_score(y_val, fold_preds, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_preds = fold_preds.copy()

    return best_preds


def train_hopfield_models(
    *,
    config: HopfieldConfig | None = None,
    detectors: Sequence[str] = DETECTORS,
    prefix_len: int | None = None,
    output_dir: Path = DEFAULT_MODEL_DIR,
) -> HopfieldTrainingSummary:
    """Train one Hopfield model per detector via stratified K-fold CV.

    If ``prefix_len`` is set, traces are truncated to that many steps before
    training/inference, simulating early detection.
    """
    cfg = config or HopfieldConfig()
    backend = _load_hopfield_backend()
    np = backend["np"]
    torch = backend["torch"]
    StratifiedKFold = backend["StratifiedKFold"]
    f1_score = backend["f1_score"]
    precision_score = backend["precision_score"]
    recall_score = backend["recall_score"]
    confusion_matrix = backend["confusion_matrix"]

    device = _resolve_device(cfg.device)
    traces, labels, _trace_ids, skipped = build_trace_dataset(config=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    Classifier = _build_classifier_class()
    dummy = Classifier(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_stored=cfg.n_stored,
        n_heads=cfg.n_heads,
        scaling=cfg.scaling,
        dropout=cfg.dropout,
    )
    model_params = dummy.param_count()

    metrics: dict[str, HopfieldEvaluationMetric] = {}
    mode = "full" if prefix_len is None else f"prefix-{prefix_len}"
    artifact_suffix = "" if prefix_len is None else f"_p{prefix_len}"

    for detector in detectors:
        y = np.array(
            [1.0 if label.get(detector, False) else 0.0 for label in labels],
            dtype=np.float32,
        )
        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives < cfg.n_splits or negatives < cfg.n_splits:
            raise ValueError(
                f"detector {detector} does not have enough class balance for "
                f"{cfg.n_splits}-fold CV (pos={positives}, neg={negatives})"
            )

        kf = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )
        all_preds = np.zeros_like(y)
        fold_f1s: list[float] = []

        for train_idx, val_idx in kf.split(traces, y):
            train_traces = [traces[i] for i in train_idx]
            val_traces = [traces[i] for i in val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            fold_preds = _train_one_fold(
                train_traces=train_traces,
                val_traces=val_traces,
                y_train=y_train,
                y_val=y_val,
                config=cfg,
                prefix_len=prefix_len,
                device=device,
            )
            all_preds[val_idx] = fold_preds
            fold_f1s.append(float(f1_score(y_val, fold_preds, zero_division=0)))

        precision = float(precision_score(y, all_preds, zero_division=0))
        recall = float(recall_score(y, all_preds, zero_division=0))
        f1 = float(f1_score(y, all_preds, zero_division=0))
        cm = confusion_matrix(y, all_preds, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])

        # Train final artifact with best-checkpoint tracking (s21-m01).
        # Hold out 15% stratified for validation to select the best epoch.
        # Pre-fix: last-epoch state was saved unconditionally, which degraded
        # p7 models whose val F1 peaked mid-training and decayed in late epochs.
        final_mean, final_std = _compute_normalization(traces)
        final_norm = _normalize(traces, final_mean, final_std)

        TraceDataset = _build_dataset_class()
        val_fraction = 0.15
        val_fold = StratifiedKFold(
            n_splits=max(2, round(1.0 / val_fraction)),
            shuffle=True,
            random_state=cfg.random_state,
        )
        train_idx, val_idx = next(val_fold.split(final_norm, y))
        train_norm = [final_norm[i] for i in train_idx]
        train_y = y[train_idx]
        val_norm = [final_norm[i] for i in val_idx]
        val_y = y[val_idx]

        train_ds = TraceDataset(train_norm, train_y, cfg.max_steps, prefix_len=prefix_len)
        train_loader = backend["DataLoader"](train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_ds = TraceDataset(val_norm, val_y, cfg.max_steps, prefix_len=prefix_len)
        val_loader_final = backend["DataLoader"](val_ds, batch_size=cfg.batch_size, shuffle=False)

        final_model = Classifier(
            n_features=N_FEATURES,
            d_model=cfg.d_model,
            n_stored=cfg.n_stored,
            n_heads=cfg.n_heads,
            scaling=cfg.scaling,
            dropout=cfg.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        F_mod = backend["F"]

        import copy

        best_val_f1 = -1.0
        best_state_dict = copy.deepcopy(final_model.state_dict())
        for _epoch in range(cfg.epochs):
            final_model.train()
            for x, lengths, y_batch in train_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = final_model(x, lengths)
                loss = F_mod.binary_cross_entropy_with_logits(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()

            # Evaluate on held-out validation split
            final_model.eval()
            val_preds_list: list[Any] = []
            with torch.no_grad():
                for x, lengths, _yb in val_loader_final:
                    x = x.to(device)
                    lengths = lengths.to(device)
                    logits = final_model(x, lengths)
                    preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                    val_preds_list.append(preds)
            epoch_val_f1 = float(f1_score(val_y, np.concatenate(val_preds_list), zero_division=0))
            if epoch_val_f1 > best_val_f1:
                best_val_f1 = epoch_val_f1
                best_state_dict = copy.deepcopy(final_model.state_dict())

        artifact_path = output_dir / f"{detector}{artifact_suffix}.pt"
        torch.save(
            {
                "detector": detector,
                "mode": mode,
                "prefix_len": prefix_len,
                "state_dict": best_state_dict,
                "mean": final_mean.tolist(),
                "std": final_std.tolist(),
                "config": _config_to_dict(cfg),
                "n_features": N_FEATURES,
                "model_params": model_params,
            },
            artifact_path,
        )

        metrics[detector] = HopfieldEvaluationMetric(
            detector=detector,
            f1=f1,
            precision=precision,
            recall=recall,
            positives=positives,
            negatives=negatives,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            artifact_path=str(artifact_path),
            f1_std=float(np.std(fold_f1s)),
            prefix_len=prefix_len,
        )

    summary = HopfieldTrainingSummary(
        trace_count=len(traces),
        skipped_traces=skipped,
        feature_count=N_FEATURES,
        model_params=model_params,
        mode=mode,
        metrics=metrics,
        output_dir=str(output_dir),
        device=str(device),
    )
    summary_name = f"summary_{mode}.json"
    (output_dir / summary_name).write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    return summary


def train_all_cutoffs(
    *,
    config: HopfieldConfig | None = None,
    cutoffs: Sequence[int] = DEFAULT_PREFIX_CUTOFFS,
    output_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, HopfieldTrainingSummary]:
    """Train Hopfield models at every prefix cutoff plus the full-trace baseline."""
    cfg = config or HopfieldConfig()
    summaries: dict[str, HopfieldTrainingSummary] = {}

    summaries["full"] = train_hopfield_models(config=cfg, prefix_len=None, output_dir=output_dir)
    for cutoff in cutoffs:
        summaries[f"prefix-{cutoff}"] = train_hopfield_models(
            config=cfg, prefix_len=cutoff, output_dir=output_dir
        )

    return summaries


# ============================================================
# Inference
# ============================================================


def load_hopfield_artifact(
    detector: str,
    *,
    prefix_len: int | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    map_location: str | None = None,
) -> Mapping[str, Any]:
    """Load a saved Hopfield artifact for a detector and optional prefix cutoff."""
    backend = _load_hopfield_backend()
    torch = backend["torch"]
    suffix = "" if prefix_len is None else f"_p{prefix_len}"
    artifact_path = model_dir / f"{detector}{suffix}.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Hopfield artifact not found: {artifact_path}")
    return cast(
        Mapping[str, Any],
        torch.load(artifact_path, map_location=map_location or "cpu", weights_only=False),
    )


def _trace_to_tensor(
    steps: Sequence[Mapping[str, Any]],
    *,
    mean: Any,
    std: Any,
    max_steps: int,
    prefix_len: int | None,
) -> tuple[Any, int]:
    backend = _load_hopfield_backend()
    np = backend["np"]
    torch = backend["torch"]

    rows = [_snapshot_row(step) for step in steps]
    matrix = np.array(rows, dtype=np.float32)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    normalized = (matrix - mean_arr) / std_arr

    actual_len = len(normalized)
    if prefix_len is not None:
        use_len = min(prefix_len, actual_len)
    else:
        use_len = min(actual_len, max_steps)
    padded = np.zeros((max_steps, normalized.shape[1]), dtype=np.float32)
    padded[:use_len] = normalized[:use_len]
    return torch.tensor(padded, dtype=torch.float32).unsqueeze(0), use_len


def predict(
    steps: Sequence[Mapping[str, Any]],
    detector: str,
    *,
    prefix_len: int | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_energy: bool = True,
) -> HopfieldPrediction:
    """Run a saved Hopfield model on a trace, optionally truncated to a prefix.

    Convenience wrapper that loads the artifact and runs a single prediction.
    For evaluation loops over many traces, prefer :func:`load_predictor`
    which loads the model once and returns a fast per-trace callable.
    """
    return load_predictor(
        detector,
        prefix_len=prefix_len,
        model_dir=model_dir,
        include_energy=include_energy,
    )(steps)


def load_predictor(
    detector: str,
    *,
    prefix_len: int | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    include_energy: bool = True,
) -> Callable[[Sequence[Mapping[str, Any]]], HopfieldPrediction]:
    """Load a saved Hopfield model once and return a cached per-trace predictor.

    The returned callable performs only the tensor conversion and forward
    pass on each invocation; the artifact load, classifier construction, and
    ``state_dict`` load happen exactly once when this factory is called. Use
    this in evaluation loops over thousands of traces where re-loading the
    artifact every call would dominate runtime.
    """
    backend = _load_hopfield_backend()
    torch = backend["torch"]
    artifact = load_hopfield_artifact(detector, prefix_len=prefix_len, model_dir=model_dir)

    cfg_dict = artifact["config"]
    Classifier = _build_classifier_class()
    model = Classifier(
        n_features=int(artifact["n_features"]),
        d_model=int(cfg_dict["d_model"]),
        n_stored=int(cfg_dict["n_stored"]),
        n_heads=int(cfg_dict["n_heads"]),
        scaling=float(cfg_dict["scaling"]),
        dropout=float(cfg_dict["dropout"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    mean = artifact["mean"]
    std = artifact["std"]
    max_steps = int(cfg_dict["max_steps"])
    captured_prefix_len = prefix_len
    capture_energy = include_energy

    def predict_one(steps: Sequence[Mapping[str, Any]]) -> HopfieldPrediction:
        x, length = _trace_to_tensor(
            steps,
            mean=mean,
            std=std,
            max_steps=max_steps,
            prefix_len=captured_prefix_len,
        )
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.no_grad():
            logits = model(x, lengths)
            proba = float(torch.sigmoid(logits).item())
            energy_value: float | None = None
            if capture_energy:
                energy_value = float(model.get_energy(x, lengths).item())
        return HopfieldPrediction(
            detected=bool(proba >= 0.5),
            probability=proba,
            energy=energy_value,
        )

    return predict_one


# ============================================================
# CLI
# ============================================================


def _config_to_dict(config: HopfieldConfig) -> dict[str, Any]:
    data = asdict(config)
    data["corpus_root"] = str(config.corpus_root)
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "prefix", "all"),
        default="all",
        help="Training mode: full-trace only, prefix-only, or all (default)",
    )
    parser.add_argument(
        "--prefix",
        type=int,
        nargs="*",
        default=list(DEFAULT_PREFIX_CUTOFFS),
        help="Prefix cutoffs to train (default: 3 5 7)",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = HopfieldConfig(
        corpus_root=args.corpus_root,
        epochs=args.epochs,
        n_splits=args.n_splits,
        device=args.device,
    )

    if args.mode in ("full", "all"):
        print(f"[hopfield] Training full-trace models (epochs={cfg.epochs})...")
        full = train_hopfield_models(config=cfg, prefix_len=None, output_dir=args.output_dir)
        print(json.dumps(full.to_dict(), indent=2))

    if args.mode in ("prefix", "all"):
        for cutoff in args.prefix:
            print(f"[hopfield] Training prefix-{cutoff} models (epochs={cfg.epochs})...")
            partial = train_hopfield_models(
                config=cfg, prefix_len=cutoff, output_dir=args.output_dir
            )
            print(json.dumps(partial.to_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
