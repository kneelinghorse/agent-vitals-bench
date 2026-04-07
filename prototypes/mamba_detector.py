"""Mamba SSM behavioral classifier prototype for bench-side validation.

A tiny pure-PyTorch Mamba (Selective State Space) model that classifies
behavioral failure modes from raw trace sequences with zero feature engineering.

Architecture: 3 MambaBlocks, d_model=64, d_state=16, expand=2 (~97K params).
Supports both per-detector binary classification and a single multi-label model
that scores all 5 detectors simultaneously.

Dependencies are gated: torch, scikit-learn, and numpy are loaded lazily via
``_load_mamba_backend()``. Install via ``pip install -e .[mamba]`` or use the
sibling ``tda-experiment`` Python 3.12 venv which already has the stack.
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
DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "mamba_models"


class MissingMambaDependencyError(RuntimeError):
    """Raised when optional Mamba dependencies (torch, sklearn, numpy) are missing."""


@dataclass(frozen=True)
class MambaConfig:
    """Tunable parameters for the Mamba detector prototype."""

    corpus_root: Path = field(default_factory=lambda: DEFAULT_CORPUS_ROOT)
    min_confidence: float = 0.8
    min_steps: int = 5
    max_steps: int = 20
    d_model: int = 64
    n_layers: int = 3
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_splits: int = 5
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    random_state: int = 42
    device: str | None = None  # auto-detect if None


@dataclass(frozen=True)
class MambaEvaluationMetric:
    """Per-detector evaluation metrics."""

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


@dataclass(frozen=True)
class MambaTrainingSummary:
    """Aggregate training summary for a Mamba run."""

    trace_count: int
    skipped_traces: int
    feature_count: int
    model_params: int
    mode: str  # "binary" or "multilabel"
    metrics: dict[str, MambaEvaluationMetric]
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
class MambaPrediction:
    """Single prediction output."""

    detected: bool
    probability: float


@lru_cache(maxsize=1)
def _load_mamba_backend() -> dict[str, Any]:
    """Lazy import of torch, sklearn, and numpy. Cached after first call."""
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise MissingMambaDependencyError(
            "Mamba dependencies are missing. Install `agent-vitals-bench[mamba]` "
            "or use the sibling tda-experiment Python 3.12 environment."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "StratifiedKFold": StratifiedKFold,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "confusion_matrix": confusion_matrix,
    }


def _resolve_device(device_pref: str | None) -> Any:
    """Pick MPS if available on Apple Silicon, else CPU."""
    backend = _load_mamba_backend()
    torch = backend["torch"]
    if device_pref is not None:
        return torch.device(device_pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Model architecture (built lazily inside factory functions)
# ============================================================


def _build_mamba_block_class() -> Any:
    """Build the MambaBlock class lazily once torch is available."""
    backend = _load_mamba_backend()
    nn = backend["nn"]
    F = backend["F"]
    torch = backend["torch"]

    class MambaBlock(nn.Module):  # type: ignore[misc, name-defined]
        """Pure-PyTorch Mamba block (no CUDA kernels). Based on mamba-minimal."""

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = d_model * expand

            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )
            self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
            self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

            A = (
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.d_inner, -1)
            )
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x: Any) -> Any:
            residual = x
            x = self.norm(x)
            _batch, seq_len, _ = x.shape

            xz = self.in_proj(x)
            x_part, z = xz.chunk(2, dim=-1)

            x_conv = x_part.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :seq_len]
            x_conv = x_conv.transpose(1, 2)
            x_conv = F.silu(x_conv)

            x_proj = self.x_proj(x_conv)
            dt = x_proj[:, :, :1]
            B = x_proj[:, :, 1 : 1 + self.d_state]
            C = x_proj[:, :, 1 + self.d_state :]

            dt = F.softplus(self.dt_proj(dt))
            A = -torch.exp(self.A_log)
            y = self._selective_scan(x_conv, dt, A, B, C)

            y = y * F.silu(z)
            y = self.out_proj(y)
            return y + residual

        def _selective_scan(self, x: Any, dt: Any, A: Any, B: Any, C: Any) -> Any:
            batch, seq_len, d_inner = x.shape
            d_state = self.d_state
            h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(seq_len):
                dA = torch.exp(dt[:, t].unsqueeze(-1) * A.unsqueeze(0))
                dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
                h = h * dA + x[:, t].unsqueeze(-1) * dB
                y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
                y = y + x[:, t] * self.D
                outputs.append(y)
            return torch.stack(outputs, dim=1)

    return MambaBlock


def _build_classifier_class(n_classes: int) -> Any:
    """Build a MambaClassifier with the given output dimension."""
    backend = _load_mamba_backend()
    nn = backend["nn"]
    MambaBlock = _build_mamba_block_class()

    class MambaClassifier(nn.Module):  # type: ignore[misc, name-defined]
        """Mamba-based trace classifier."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            n_layers: int = 3,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.layers = nn.ModuleList(
                [
                    MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                    for _ in range(n_layers)
                ]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, n_classes),
            )
            self._n_classes = n_classes

        def forward(self, x: Any, lengths: Any | None = None) -> Any:
            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)

            if lengths is not None:
                idx = (lengths - 1).long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.shape[-1])
                pooled = x.gather(1, idx).squeeze(1)
            else:
                pooled = x[:, -1, :]

            logits = self.head(pooled)
            if self._n_classes == 1:
                return logits.squeeze(-1)
            return logits

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

    return MambaClassifier


def _build_dataset_class() -> Any:
    """Build the TraceDataset class for variable-length trace sequences."""
    backend = _load_mamba_backend()
    Dataset = backend["Dataset"]
    torch = backend["torch"]
    np = backend["np"]

    class TraceDataset(Dataset):  # type: ignore[misc, valid-type]
        """Pads/truncates traces to a fixed length and exposes (x, length, y)."""

        def __init__(self, traces: list[Any], labels: Any, max_len: int) -> None:
            self.traces = traces
            self.labels = labels
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.traces)

        def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
            trace = self.traces[idx]
            label = self.labels[idx]
            length = min(len(trace), self.max_len)

            padded = np.zeros((self.max_len, trace.shape[1]), dtype=np.float32)
            padded[:length] = trace[:length]

            return (
                torch.tensor(padded, dtype=torch.float32),
                torch.tensor(length, dtype=torch.long),
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


def load_manifest_entries(*, config: MambaConfig | None = None) -> list[dict[str, Any]]:
    """Load manifest entries that meet the minimum confidence threshold."""
    cfg = config or MambaConfig()
    manifest_path = cfg.corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return [
        entry
        for entry in manifest
        if float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0) >= cfg.min_confidence
    ]


def _load_trace_steps(entry: Mapping[str, Any], config: MambaConfig) -> list[dict[str, Any]] | None:
    trace_path = config.corpus_root / str(entry["path"])
    if not trace_path.exists():
        return None
    steps = json.loads(trace_path.read_text())
    if not isinstance(steps, list):
        return None
    return steps


def build_trace_dataset(
    *,
    config: MambaConfig | None = None,
    manifest_entries: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Any], list[dict[str, bool]], list[str], int]:
    """Load traces from the corpus into per-trace numpy matrices."""
    cfg = config or MambaConfig()
    backend = _load_mamba_backend()
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
    """Compute global mean/std across all traces for normalization."""
    backend = _load_mamba_backend()
    np = backend["np"]
    all_steps = np.vstack(list(traces))
    return all_steps.mean(axis=0), all_steps.std(axis=0) + 1e-8


def _normalize(traces: Sequence[Any], mean: Any, std: Any) -> list[Any]:
    return [(t - mean) / std for t in traces]


# ============================================================
# Training (binary mode)
# ============================================================


def _train_one_fold(
    *,
    train_traces: list[Any],
    val_traces: list[Any],
    y_train: Any,
    y_val: Any,
    config: MambaConfig,
    n_classes: int,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    """Train a single fold and return (best_preds, val_labels, mean, std)."""
    backend = _load_mamba_backend()
    F = backend["F"]
    torch = backend["torch"]
    DataLoader = backend["DataLoader"]
    f1_score = backend["f1_score"]

    mean, std = _compute_normalization(train_traces)
    train_norm = _normalize(train_traces, mean, std)
    val_norm = _normalize(val_traces, mean, std)

    TraceDataset = _build_dataset_class()
    train_ds = TraceDataset(train_norm, y_train, config.max_steps)
    val_ds = TraceDataset(val_norm, y_val, config.max_steps)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    Classifier = _build_classifier_class(n_classes)
    model = Classifier(
        n_features=N_FEATURES,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
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
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        all_preds: list[Any] = []
        with torch.no_grad():
            for x, lengths, _y in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                logits = model(x, lengths)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                all_preds.append(preds)
        fold_preds = backend["np"].concatenate(all_preds, axis=0)

        if n_classes == 1:
            current_f1 = float(f1_score(y_val, fold_preds, zero_division=0))
        else:
            # Macro F1 for multi-label
            current_f1 = float(f1_score(y_val, fold_preds, average="macro", zero_division=0))

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_preds = fold_preds

    return best_preds, model, mean, std


def train_binary_models(
    *,
    config: MambaConfig | None = None,
    detectors: Sequence[str] = DETECTORS,
    output_dir: Path = DEFAULT_MODEL_DIR,
) -> MambaTrainingSummary:
    """Train one Mamba model per detector via stratified K-fold CV.

    Final artifacts are trained on the full dataset after CV scoring.
    """
    cfg = config or MambaConfig()
    backend = _load_mamba_backend()
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

    Classifier = _build_classifier_class(1)
    dummy = Classifier(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    )
    model_params = dummy.param_count()

    metrics: dict[str, MambaEvaluationMetric] = {}

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

            fold_preds, _model, _mean, _std = _train_one_fold(
                train_traces=train_traces,
                val_traces=val_traces,
                y_train=y_train,
                y_val=y_val,
                config=cfg,
                n_classes=1,
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

        # Train final model on full dataset for the artifact
        final_mean, final_std = _compute_normalization(traces)
        final_norm = _normalize(traces, final_mean, final_std)

        TraceDataset = _build_dataset_class()
        final_ds = TraceDataset(final_norm, y, cfg.max_steps)
        final_loader = backend["DataLoader"](final_ds, batch_size=cfg.batch_size, shuffle=True)

        final_model = Classifier(
            n_features=N_FEATURES,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
        ).to(device)
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        F_mod = backend["F"]
        for _epoch in range(cfg.epochs):
            final_model.train()
            for x, lengths, y_batch in final_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = final_model(x, lengths)
                loss = F_mod.binary_cross_entropy_with_logits(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

        artifact_path = output_dir / f"{detector}.pt"
        torch.save(
            {
                "detector": detector,
                "mode": "binary",
                "state_dict": final_model.state_dict(),
                "mean": final_mean.tolist(),
                "std": final_std.tolist(),
                "config": _config_to_dict(cfg),
                "n_features": N_FEATURES,
                "model_params": model_params,
            },
            artifact_path,
        )

        metrics[detector] = MambaEvaluationMetric(
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
        )

    summary = MambaTrainingSummary(
        trace_count=len(traces),
        skipped_traces=skipped,
        feature_count=N_FEATURES,
        model_params=model_params,
        mode="binary",
        metrics=metrics,
        output_dir=str(output_dir),
        device=str(device),
    )
    (output_dir / "summary_binary.json").write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    return summary


# ============================================================
# Multi-label training
# ============================================================


def train_multilabel_model(
    *,
    config: MambaConfig | None = None,
    output_dir: Path = DEFAULT_MODEL_DIR,
) -> MambaTrainingSummary:
    """Train a single Mamba model that scores all 5 detectors at once."""
    cfg = config or MambaConfig()
    backend = _load_mamba_backend()
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

    n_classes = len(DETECTORS)
    Classifier = _build_classifier_class(n_classes)
    dummy = Classifier(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    )
    model_params = dummy.param_count()

    y_multi = np.array(
        [[1.0 if label.get(d, False) else 0.0 for d in DETECTORS] for label in labels],
        dtype=np.float32,
    )
    # Stratification key combines all 5 labels into a single integer
    y_strat = np.array(
        [int("".join(str(int(v)) for v in row), 2) for row in y_multi],
        dtype=np.int64,
    )

    # Drop strata with a single sample (StratifiedKFold cannot handle them)
    from collections import Counter

    counts = Counter(y_strat.tolist())
    too_small = {k for k, v in counts.items() if v < cfg.n_splits}
    keep_mask = np.array([k not in too_small for k in y_strat.tolist()], dtype=bool)
    if not keep_mask.all():
        traces = [t for t, keep in zip(traces, keep_mask) if keep]
        y_multi = y_multi[keep_mask]
        y_strat = y_strat[keep_mask]

    kf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )
    all_preds = np.zeros_like(y_multi)

    for train_idx, val_idx in kf.split(traces, y_strat):
        train_traces = [traces[i] for i in train_idx]
        val_traces = [traces[i] for i in val_idx]
        y_train = y_multi[train_idx]
        y_val = y_multi[val_idx]

        fold_preds, _model, _mean, _std = _train_one_fold(
            train_traces=train_traces,
            val_traces=val_traces,
            y_train=y_train,
            y_val=y_val,
            config=cfg,
            n_classes=n_classes,
            device=device,
        )
        all_preds[val_idx] = fold_preds

    metrics: dict[str, MambaEvaluationMetric] = {}

    # Train final model on full dataset
    final_mean, final_std = _compute_normalization(traces)
    final_norm = _normalize(traces, final_mean, final_std)

    TraceDataset = _build_dataset_class()
    final_ds = TraceDataset(final_norm, y_multi, cfg.max_steps)
    final_loader = backend["DataLoader"](final_ds, batch_size=cfg.batch_size, shuffle=True)

    final_model = Classifier(
        n_features=N_FEATURES,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    ).to(device)
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    F_mod = backend["F"]
    for _epoch in range(cfg.epochs):
        final_model.train()
        for x, lengths, y_batch in final_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = final_model(x, lengths)
            loss = F_mod.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    artifact_path = output_dir / "multilabel.pt"
    torch.save(
        {
            "detectors": list(DETECTORS),
            "mode": "multilabel",
            "state_dict": final_model.state_dict(),
            "mean": final_mean.tolist(),
            "std": final_std.tolist(),
            "config": _config_to_dict(cfg),
            "n_features": N_FEATURES,
            "n_classes": n_classes,
            "model_params": model_params,
        },
        artifact_path,
    )

    for i, detector in enumerate(DETECTORS):
        y_true = y_multi[:, i]
        y_pred = all_preds[:, i]
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)

        metrics[detector] = MambaEvaluationMetric(
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
        )

    summary = MambaTrainingSummary(
        trace_count=len(traces),
        skipped_traces=skipped,
        feature_count=N_FEATURES,
        model_params=model_params,
        mode="multilabel",
        metrics=metrics,
        output_dir=str(output_dir),
        device=str(device),
    )
    (output_dir / "summary_multilabel.json").write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n"
    )
    return summary


# ============================================================
# Inference
# ============================================================


def load_mamba_artifact(
    detector_or_mode: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    map_location: str | None = None,
) -> Mapping[str, Any]:
    """Load a saved Mamba artifact (binary detector name or 'multilabel')."""
    backend = _load_mamba_backend()
    torch = backend["torch"]
    artifact_path = model_dir / f"{detector_or_mode}.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Mamba artifact not found: {artifact_path}")
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
) -> tuple[Any, int]:
    """Convert raw trace steps to a normalized tensor."""
    backend = _load_mamba_backend()
    np = backend["np"]
    torch = backend["torch"]

    rows = [_snapshot_row(step) for step in steps]
    matrix = np.array(rows, dtype=np.float32)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    normalized = (matrix - mean_arr) / std_arr

    length = min(len(normalized), max_steps)
    padded = np.zeros((max_steps, normalized.shape[1]), dtype=np.float32)
    padded[:length] = normalized[:length]
    return torch.tensor(padded, dtype=torch.float32).unsqueeze(0), length


def predict_binary(
    steps: Sequence[Mapping[str, Any]],
    detector: str,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> MambaPrediction:
    """Run a saved binary Mamba model on a trace."""
    backend = _load_mamba_backend()
    torch = backend["torch"]
    artifact = load_mamba_artifact(detector, model_dir=model_dir)
    if artifact.get("mode") != "binary":
        raise ValueError(f"Artifact at {detector}.pt is not a binary model")

    cfg_dict = artifact["config"]
    Classifier = _build_classifier_class(1)
    model = Classifier(
        n_features=int(artifact["n_features"]),
        d_model=int(cfg_dict["d_model"]),
        n_layers=int(cfg_dict["n_layers"]),
        d_state=int(cfg_dict["d_state"]),
        d_conv=int(cfg_dict["d_conv"]),
        expand=int(cfg_dict["expand"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    x, length = _trace_to_tensor(
        steps,
        mean=artifact["mean"],
        std=artifact["std"],
        max_steps=int(cfg_dict["max_steps"]),
    )
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths)
        proba = float(torch.sigmoid(logits).item())
    return MambaPrediction(detected=bool(proba >= 0.5), probability=proba)


def predict_multilabel(
    steps: Sequence[Mapping[str, Any]],
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, MambaPrediction]:
    """Run the saved multi-label Mamba model on a trace.

    Convenience wrapper that loads the artifact and runs a single prediction.
    For evaluation loops over many traces, prefer
    :func:`load_multilabel_predictor` which loads the model once and returns
    a fast per-trace predictor.
    """
    return load_multilabel_predictor(model_dir=model_dir)(steps)


def load_multilabel_predictor(
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, MambaPrediction]]:
    """Load the multi-label Mamba model once and return a cached predictor.

    The returned callable performs only the per-trace tensor conversion and
    forward pass on each invocation; the artifact load, classifier
    construction, and ``state_dict`` load happen exactly once when this
    factory is called. Use this in evaluation loops over thousands of traces
    where re-loading the artifact every call would dominate runtime.
    """
    backend = _load_mamba_backend()
    torch = backend["torch"]
    artifact = load_mamba_artifact("multilabel", model_dir=model_dir)
    if artifact.get("mode") != "multilabel":
        raise ValueError("Artifact at multilabel.pt is not a multi-label model")

    cfg_dict = artifact["config"]
    n_classes = int(artifact["n_classes"])
    Classifier = _build_classifier_class(n_classes)
    model = Classifier(
        n_features=int(artifact["n_features"]),
        d_model=int(cfg_dict["d_model"]),
        n_layers=int(cfg_dict["n_layers"]),
        d_state=int(cfg_dict["d_state"]),
        d_conv=int(cfg_dict["d_conv"]),
        expand=int(cfg_dict["expand"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    detectors = list(artifact["detectors"])
    mean = artifact["mean"]
    std = artifact["std"]
    max_steps = int(cfg_dict["max_steps"])

    def predict(steps: Sequence[Mapping[str, Any]]) -> dict[str, MambaPrediction]:
        x, length = _trace_to_tensor(steps, mean=mean, std=std, max_steps=max_steps)
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.no_grad():
            logits = model(x, lengths)
            probas = torch.sigmoid(logits).squeeze(0).tolist()
        return {
            det: MambaPrediction(detected=bool(p >= 0.5), probability=float(p))
            for det, p in zip(detectors, probas)
        }

    return predict


# ============================================================
# CLI
# ============================================================


def _config_to_dict(config: MambaConfig) -> dict[str, Any]:
    data = asdict(config)
    data["corpus_root"] = str(config.corpus_root)
    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("binary", "multilabel", "both"),
        default="both",
        help="Training mode (default: both)",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Path to the corpus root (default: bench corpus/v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to save model artifacts",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/mps/cuda)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = MambaConfig(
        corpus_root=args.corpus_root,
        epochs=args.epochs,
        n_splits=args.n_splits,
        device=args.device,
    )

    summaries = []
    if args.mode in ("binary", "both"):
        print(f"[mamba] Training binary models (epochs={cfg.epochs})...")
        binary_summary = train_binary_models(config=cfg, output_dir=args.output_dir)
        print(json.dumps(binary_summary.to_dict(), indent=2))
        summaries.append(binary_summary)

    if args.mode in ("multilabel", "both"):
        print(f"[mamba] Training multi-label model (epochs={cfg.epochs})...")
        multi_summary = train_multilabel_model(config=cfg, output_dir=args.output_dir)
        print(json.dumps(multi_summary.to_dict(), indent=2))
        summaries.append(multi_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
