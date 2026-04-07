"""Partial-trace evaluation harness for early-detection benchmarking.

The standard evaluator scores traces in their entirety. This module wraps the
evaluator pipeline so that traces can be truncated to a fixed step cutoff
before detection runs, simulating early-detection scenarios where only the
first N steps of a trace are visible.

Cutoffs are arbitrary positive integers. ``None`` represents the full trace
(no truncation). The harness accepts a pluggable ``TracePredictor`` callable
so the same infrastructure can score handcrafted detectors (via
``replay_trace``) or learned prototypes (Mamba, Hopfield, TDA).

Example
-------

    from evaluator.partial_trace import (
        PartialTraceConfig,
        evaluate_partial_traces,
    )

    config = PartialTraceConfig(cutoffs=(3, 5, 7, None))
    result = evaluate_partial_traces(corpus_version="v1", config=config)
    for cutoff, cutoff_metrics in result.per_cutoff.items():
        print(cutoff, cutoff_metrics.detector_metrics["loop"].f1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from agent_vitals import VitalsConfig
from agent_vitals.schema import VitalsSnapshot

from evaluator.metrics import DetectorMetrics
from evaluator.runner import (
    DETECTORS,
    TraceResult,
    load_manifest,
    load_trace,
    replay_trace,
    resolve_workflow_type,
)


class TracePredictor(Protocol):
    """A callable that maps a trace prefix to detector predictions.

    Used by the partial-trace harness to plug in either handcrafted detectors
    (via ``replay_trace``) or learned prototypes (Mamba, Hopfield, TDA).
    """

    def __call__(self, snapshots: Sequence[VitalsSnapshot]) -> Mapping[str, bool]: ...


@dataclass(frozen=True)
class PartialTraceConfig:
    """Configuration for partial-trace evaluation."""

    # ``None`` represents the full trace (no truncation).
    cutoffs: tuple[int | None, ...] = (3, 5, 7, None)
    min_confidence: float = 0.8
    # Minimum step count required for a trace to participate. Traces shorter
    # than this are skipped entirely (not just at one cutoff).
    min_steps: int = 1
    # If True, a trace is skipped at cutoffs longer than its actual length.
    # If False, the full trace is used at those cutoffs (matching the
    # full-trace baseline behavior).
    skip_short_traces: bool = False
    # Optional framework profile name passed to ``VitalsConfig.for_framework``
    profile: str | None = None
    # Optional manifest filter: only include traces whose path starts with
    # one of these prefixes. Use to focus on a single detector's corpus.
    path_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.cutoffs:
            raise ValueError("cutoffs must contain at least one value")
        for cutoff in self.cutoffs:
            if cutoff is not None and cutoff < 1:
                raise ValueError(f"cutoff must be a positive integer or None, got {cutoff}")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")


@dataclass
class CutoffMetrics:
    """Per-cutoff evaluation results."""

    cutoff: int | None
    trace_count: int
    skipped_short: int
    skipped_min_steps: int
    detector_metrics: dict[str, DetectorMetrics]
    detectors_evaluated: tuple[str, ...]
    trace_results: list[TraceResult] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "full" if self.cutoff is None else f"prefix-{self.cutoff}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "label": self.label,
            "trace_count": self.trace_count,
            "skipped_short": self.skipped_short,
            "skipped_min_steps": self.skipped_min_steps,
            "detectors_evaluated": list(self.detectors_evaluated),
            "metrics": {
                detector: metric.as_dict() for detector, metric in self.detector_metrics.items()
            },
        }


@dataclass
class PartialTraceEvaluation:
    """Aggregate evaluation across all cutoffs."""

    corpus_version: str
    cutoffs: tuple[int | None, ...]
    per_cutoff: dict[int | None, CutoffMetrics]
    config: PartialTraceConfig
    predictor_label: str

    def metric(self, cutoff: int | None, detector: str) -> DetectorMetrics:
        return self.per_cutoff[cutoff].detector_metrics[detector]

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "predictor": self.predictor_label,
            "config": {
                "cutoffs": [c if c is not None else None for c in self.config.cutoffs],
                "min_confidence": self.config.min_confidence,
                "min_steps": self.config.min_steps,
                "skip_short_traces": self.config.skip_short_traces,
                "profile": self.config.profile,
                "path_prefixes": list(self.config.path_prefixes),
            },
            "per_cutoff": [self.per_cutoff[cutoff].as_dict() for cutoff in self.cutoffs],
        }


def truncate_trace(
    snapshots: Sequence[VitalsSnapshot],
    cutoff: int | None,
) -> list[VitalsSnapshot]:
    """Return a trace truncated to ``cutoff`` steps.

    A ``cutoff`` of ``None`` returns the full trace. A ``cutoff`` longer than
    the trace returns the full trace (no padding). A ``cutoff`` of 0 raises
    ``ValueError`` to avoid silently producing empty traces.
    """
    if cutoff is None:
        return list(snapshots)
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1 or None, got {cutoff}")
    return list(snapshots[:cutoff])


def make_handcrafted_predictor(
    *,
    config: VitalsConfig | None = None,
    profile: str | None = None,
) -> TracePredictor:
    """Build a TracePredictor backed by the handcrafted ``replay_trace`` pipeline.

    The returned callable resolves the workflow type from each trace's first
    snapshot and runs the standard handcrafted detectors. This is the default
    predictor used by ``evaluate_partial_traces`` when none is supplied.
    """
    base_config = config or VitalsConfig.from_yaml()
    effective_config = base_config.for_framework(profile) if profile else base_config

    def predictor(snapshots: Sequence[VitalsSnapshot]) -> dict[str, bool]:
        if not snapshots:
            return {detector: False for detector in DETECTORS}
        mission_id = snapshots[0].mission_id
        # The trace_id isn't available here; resolve from mission_id alone.
        # ``resolve_workflow_type`` only inspects the combined string for
        # ``.bc.``/``.rc.`` markers so this is sufficient.
        wf_type = resolve_workflow_type("", mission_id)
        return replay_trace(list(snapshots), effective_config, workflow_type=wf_type)

    return predictor


def _filter_manifest(
    manifest: Sequence[Mapping[str, Any]],
    *,
    config: PartialTraceConfig,
) -> list[Mapping[str, Any]]:
    """Apply confidence and path-prefix filters to manifest entries."""
    filtered: list[Mapping[str, Any]] = []
    for entry in manifest:
        meta_confidence = float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0)
        if meta_confidence < config.min_confidence:
            continue
        if config.profile:
            entry_framework = entry.get("metadata", {}).get("framework")
            if entry_framework and entry_framework != config.profile:
                continue
        if config.path_prefixes:
            path = str(entry.get("path", ""))
            if not any(path.startswith(prefix) for prefix in config.path_prefixes):
                continue
        filtered.append(entry)
    return filtered


def evaluate_at_cutoff(
    *,
    corpus_version: str = "v1",
    cutoff: int | None,
    config: PartialTraceConfig | None = None,
    predictor: TracePredictor | None = None,
    detectors: tuple[str, ...] = DETECTORS,
    manifest: Sequence[Mapping[str, Any]] | None = None,
    snapshots_by_path: Mapping[str, list[VitalsSnapshot]] | None = None,
) -> CutoffMetrics:
    """Evaluate a single cutoff against the corpus.

    ``manifest`` and ``snapshots_by_path`` can be supplied to avoid reloading
    files when running multiple cutoffs back-to-back (see
    ``evaluate_partial_traces``).
    """
    cfg = config or PartialTraceConfig()
    target_detectors = detectors
    fn_predictor = predictor or make_handcrafted_predictor(profile=cfg.profile)

    entries = (
        list(manifest)
        if manifest is not None
        else _filter_manifest(load_manifest(corpus_version), config=cfg)
    )

    detector_metrics = {
        detector: DetectorMetrics(detector=detector) for detector in target_detectors
    }
    trace_results: list[TraceResult] = []
    skipped_short = 0
    skipped_min_steps = 0

    for entry in entries:
        trace_id = str(entry["trace_id"])
        labels = dict(entry.get("labels", {}))
        path = str(entry["path"])

        if snapshots_by_path is not None and path in snapshots_by_path:
            snapshots = snapshots_by_path[path]
        else:
            snapshots = load_trace(corpus_version, path)

        if len(snapshots) < cfg.min_steps:
            skipped_min_steps += 1
            continue

        if cutoff is not None and len(snapshots) < cutoff and cfg.skip_short_traces:
            skipped_short += 1
            continue

        truncated = truncate_trace(snapshots, cutoff)
        if not truncated:
            skipped_short += 1
            continue

        predictions_raw = fn_predictor(truncated)
        predictions = {
            detector: bool(predictions_raw.get(detector, False)) for detector in target_detectors
        }

        meta_confidence = float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0)
        result = TraceResult(
            trace_id=trace_id,
            labels=labels,
            predictions=predictions,
            confidence=meta_confidence,
        )
        trace_results.append(result)

        for detector in target_detectors:
            expected = bool(labels.get(detector, False))
            detector_metrics[detector].record(
                predicted=predictions[detector],
                expected=expected,
            )

    return CutoffMetrics(
        cutoff=cutoff,
        trace_count=len(trace_results),
        skipped_short=skipped_short,
        skipped_min_steps=skipped_min_steps,
        detector_metrics=detector_metrics,
        detectors_evaluated=target_detectors,
        trace_results=trace_results,
    )


def evaluate_partial_traces(
    *,
    corpus_version: str = "v1",
    config: PartialTraceConfig | None = None,
    predictor: TracePredictor | None = None,
    detectors: tuple[str, ...] = DETECTORS,
    predictor_label: str = "handcrafted",
) -> PartialTraceEvaluation:
    """Evaluate a corpus across multiple step cutoffs.

    Loads each trace once and reuses the snapshots across all cutoffs to
    avoid redundant disk reads. Returns aggregated metrics keyed by cutoff.
    """
    cfg = config or PartialTraceConfig()
    fn_predictor = predictor or make_handcrafted_predictor(profile=cfg.profile)

    manifest = _filter_manifest(load_manifest(corpus_version), config=cfg)

    snapshots_by_path: dict[str, list[VitalsSnapshot]] = {}
    for entry in manifest:
        path = str(entry["path"])
        if path not in snapshots_by_path:
            snapshots_by_path[path] = load_trace(corpus_version, path)

    per_cutoff: dict[int | None, CutoffMetrics] = {}
    for cutoff in cfg.cutoffs:
        per_cutoff[cutoff] = evaluate_at_cutoff(
            corpus_version=corpus_version,
            cutoff=cutoff,
            config=cfg,
            predictor=fn_predictor,
            detectors=detectors,
            manifest=manifest,
            snapshots_by_path=snapshots_by_path,
        )

    return PartialTraceEvaluation(
        corpus_version=corpus_version,
        cutoffs=cfg.cutoffs,
        per_cutoff=per_cutoff,
        config=cfg,
        predictor_label=predictor_label,
    )


def format_partial_trace_table(evaluation: PartialTraceEvaluation) -> str:
    """Render an F1 table for a partial-trace evaluation (one row per detector)."""
    headers = ["Detector"] + [
        ("Full" if cutoff is None else f"{cutoff} steps") for cutoff in evaluation.cutoffs
    ]
    rows: list[list[str]] = [headers]
    for detector in evaluation.per_cutoff[evaluation.cutoffs[0]].detectors_evaluated:
        row = [detector]
        for cutoff in evaluation.cutoffs:
            metric = evaluation.metric(cutoff, detector)
            row.append(f"{metric.f1:.3f}")
        rows.append(row)

    widths = [max(len(row[col]) for row in rows) for col in range(len(headers))]
    lines: list[str] = []
    for idx, row in enumerate(rows):
        cells = [cell.rjust(widths[col]) for col, cell in enumerate(row)]
        lines.append(" | ".join(cells))
        if idx == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


# Re-export for clarity
TruncationCallback = Callable[[Sequence[VitalsSnapshot], int | None], list[VitalsSnapshot]]

__all__ = [
    "DETECTORS",
    "CutoffMetrics",
    "PartialTraceConfig",
    "PartialTraceEvaluation",
    "TracePredictor",
    "TruncationCallback",
    "evaluate_at_cutoff",
    "evaluate_partial_traces",
    "format_partial_trace_table",
    "make_handcrafted_predictor",
    "truncate_trace",
]
