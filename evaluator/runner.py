"""Evaluator runner — loads corpus and replays traces through detectors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_vitals import VitalsConfig
from agent_vitals.backtest import replay_trace as _production_replay_trace
from agent_vitals.schema import VitalsSnapshot

from evaluator.metrics import DetectorMetrics, compute_metrics
from evaluator.gate import check_all_gates
from evaluator.reporter import report_results

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"

DETECTORS = ("loop", "stuck", "confabulation", "thrash", "runaway_cost")
RUNTIME_MODES = ("default", "tda")


@dataclass
class TraceResult:
    """Detection results for a single trace."""

    trace_id: str
    labels: dict[str, bool]
    predictions: dict[str, bool]
    confidence: float


@dataclass
class EvaluationResult:
    """Full evaluation results across all traces."""

    corpus_version: str
    trace_count: int
    trace_results: list[TraceResult]
    detector_metrics: dict[str, DetectorMetrics]
    gate_results: dict[str, dict[str, Any]]
    detectors_evaluated: tuple[str, ...]
    profile: str | None = None
    runtime_mode: str = "default"
    excluded_detectors: tuple[str, ...] = ()


def _detect_excluded_detectors(config: VitalsConfig) -> tuple[str, ...]:
    """Determine which detectors are intentionally disabled by profile config.

    When a detector is disabled by design (e.g., stuck with workflow_stuck_enabled=none),
    it should be excluded from the composite gate rather than counting as NO-GO.
    """
    excluded: list[str] = []
    mode = (config.workflow_stuck_enabled or "").strip().lower().replace("_", "-")
    if mode in ("none", "off", "disabled"):
        excluded.append("stuck")
    return tuple(excluded)


def load_manifest(corpus_version: str) -> list[dict[str, Any]]:
    """Load and return manifest entries for a corpus version."""
    manifest_path = CORPUS_ROOT / corpus_version / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Manifest must be a JSON array, got {type(entries).__name__}")
    return entries


def load_trace(corpus_version: str, trace_path: str) -> list[VitalsSnapshot]:
    """Load a trace file as a list of VitalsSnapshot objects."""
    full_path = CORPUS_ROOT / corpus_version / trace_path
    if not full_path.exists():
        raise FileNotFoundError(f"Trace file not found: {full_path}")

    with open(full_path) as f:
        content = f.read().strip()

    # Support both JSON array and JSONL formats
    if content.startswith("["):
        raw_entries = json.loads(content)
    else:
        raw_entries = [json.loads(line) for line in content.splitlines() if line.strip()]

    return [VitalsSnapshot.model_validate(entry) for entry in raw_entries]


def resolve_workflow_type(trace_id: str, mission_id: str | None = None) -> str:
    """Resolve workflow type from trace/mission naming conventions.

    Matches agent-vitals backtest.py resolve_workflow_type():
    - '.bc.' in name → 'build'
    - '.rc.' in name → 'research'
    - otherwise → 'unknown'
    """
    check = f"{trace_id} {mission_id or ''}"
    if ".bc." in check:
        return "build"
    if ".rc." in check:
        return "research"
    return "unknown"


def _config_for_runtime_mode(
    config: VitalsConfig | None,
    runtime_mode: str | None = None,
) -> VitalsConfig:
    """Return a config with an explicit runtime mode applied.

    Bench separates corpus/metrics/reporting from app execution semantics.
    Production replay remains canonical, and runtime mode only toggles
    whether the optional TDA runaway adjudicator is enabled.
    """
    cfg = config or VitalsConfig.from_yaml()
    if runtime_mode is None:
        return cfg
    if runtime_mode not in RUNTIME_MODES:
        allowed = ", ".join(RUNTIME_MODES)
        raise ValueError(f"runtime_mode must be one of {allowed}, got {runtime_mode!r}")
    target_tda_enabled = runtime_mode == "tda"
    if cfg.tda_enabled == target_tda_enabled:
        return cfg
    return replace(cfg, tda_enabled=target_tda_enabled)


def replay_trace(
    snapshots: list[VitalsSnapshot],
    config: VitalsConfig | None = None,
    workflow_type: str = "unknown",
    runtime_mode: str | None = None,
) -> dict[str, bool]:
    """Replay a trace through the canonical production detector pipeline.

    Bench owns corpus/labels/metrics/reporting. Agent-vitals owns replay
    semantics. This wrapper delegates to the public replay_trace() API
    (v1.16.0+) which returns the canonical five detectors directly.
    """
    if not snapshots:
        return {d: False for d in DETECTORS}

    cfg = _config_for_runtime_mode(config, runtime_mode)
    predictions = _production_replay_trace(
        snapshots,
        config=cfg,
        workflow_type=workflow_type,
    )
    return {detector: bool(predictions.get(detector, False)) for detector in DETECTORS}


def run_evaluation(
    corpus_version: str = "v1",
    detectors: tuple[str, ...] | None = None,
    min_confidence: float = 0.8,
    config: VitalsConfig | None = None,
    profile: str | None = None,
    runtime_mode: str | None = None,
) -> EvaluationResult:
    """Run full evaluation pipeline against a corpus version.

    Args:
        profile: Framework profile name (langgraph, crewai, dspy).
            When set, applies per-profile threshold overrides from
            agent-vitals thresholds.yaml via VitalsConfig.for_framework().
            Traces are also filtered to matching framework if the manifest
            entry includes a ``framework`` field.
        runtime_mode: Explicit runtime detector mode. ``default`` forces
            ``tda_enabled=False``. ``tda`` forces ``tda_enabled=True``.
            ``None`` preserves the supplied config as-is.
    """
    target_detectors = detectors or DETECTORS

    # Load base config, then apply per-profile overrides if requested
    base_config = config or VitalsConfig.from_yaml()
    effective_config = base_config.for_framework(profile) if profile else base_config
    effective_config = _config_for_runtime_mode(effective_config, runtime_mode)
    resolved_runtime_mode = "tda" if effective_config.tda_enabled else "default"

    manifest = load_manifest(corpus_version)

    trace_results: list[TraceResult] = []

    for entry in manifest:
        # Filter by confidence
        meta_confidence = entry.get("metadata", {}).get("confidence", 0.0)
        if meta_confidence < min_confidence:
            continue

        # Filter by framework profile when specified
        if profile:
            entry_framework = entry.get("metadata", {}).get("framework")
            if entry_framework and entry_framework != profile:
                continue

        trace_id = entry["trace_id"]
        labels = entry["labels"]
        trace_path = entry["path"]

        snapshots = load_trace(corpus_version, trace_path)

        # Resolve workflow type from trace naming (bc=build, rc=research)
        mission_id = snapshots[0].mission_id if snapshots else None
        wf_type = resolve_workflow_type(trace_id, mission_id)

        predictions = replay_trace(
            snapshots,
            effective_config,
            workflow_type=wf_type,
            runtime_mode=resolved_runtime_mode,
        )

        trace_results.append(
            TraceResult(
                trace_id=trace_id,
                labels=labels,
                predictions=predictions,
                confidence=meta_confidence,
            )
        )

    # Compute metrics per detector
    detector_metrics = compute_metrics(trace_results, target_detectors)

    # Detect disabled detectors for gate exclusion
    excluded = _detect_excluded_detectors(effective_config)

    # Check gates
    gate_results = check_all_gates(detector_metrics)

    # Mark excluded detectors
    for name in excluded:
        if name in gate_results:
            gate_results[name]["excluded"] = True
            gate_results[name]["status"] = "EXCLUDED"
            gate_results[name]["exclude_reason"] = "detector disabled by profile"

    return EvaluationResult(
        corpus_version=corpus_version,
        trace_count=len(trace_results),
        trace_results=trace_results,
        detector_metrics=detector_metrics,
        gate_results=gate_results,
        detectors_evaluated=target_detectors,
        profile=profile,
        runtime_mode=resolved_runtime_mode,
        excluded_detectors=excluded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Vitals Bench Evaluator")
    parser.add_argument("--corpus", default="v1", help="Corpus version (default: v1)")
    parser.add_argument(
        "--detectors",
        default="all",
        help="Comma-separated detector names or 'all'",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum trace confidence for inclusion (default: 0.8)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["langgraph", "crewai", "dspy"],
        help="Framework profile — applies per-profile threshold overrides from agent-vitals",
    )
    parser.add_argument(
        "--runtime-mode",
        default="default",
        choices=list(RUNTIME_MODES),
        help="Runtime detector mode: default keeps TDA off, tda forces the TDA adjudicator on",
    )
    args = parser.parse_args()

    detectors_arg: tuple[str, ...] | None = None
    if args.detectors != "all":
        detectors_arg = tuple(d.strip() for d in args.detectors.split(","))

    result = run_evaluation(
        corpus_version=args.corpus,
        detectors=detectors_arg,
        min_confidence=args.min_confidence,
        profile=args.profile,
        runtime_mode=args.runtime_mode,
    )

    report = report_results(result)
    print(report)

    # Exit with non-zero if any non-excluded detector failed gate
    any_failed = any(
        not g.get("passed", False)
        for g in result.gate_results.values()
        if not g.get("excluded", False)
    )
    sys.exit(1 if any_failed and result.trace_count > 0 else 0)


if __name__ == "__main__":
    main()
