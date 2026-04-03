"""Evaluator runner — loads corpus and replays traces through detectors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_vitals import VitalsConfig, detect_loop, derive_stop_signals
from agent_vitals.schema import VitalsSnapshot

from evaluator.metrics import DetectorMetrics, compute_metrics
from evaluator.gate import check_all_gates
from evaluator.reporter import report_results

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"

DETECTORS = ("loop", "stuck", "confabulation", "thrash", "runaway_cost")


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


def replay_trace(
    snapshots: list[VitalsSnapshot],
    config: VitalsConfig | None = None,
    workflow_type: str = "unknown",
) -> dict[str, bool]:
    """Replay a trace through all detectors and return predictions.

    Mirrors agent-vitals backtest.py _replay_trace() exactly:
    - Step-by-step replay through detect_loop with accumulated history
    - Stuck suppression: skip if detector_priority is confabulation or
      stuck_trigger is burn_rate_anomaly
    - Co-occurrence resolution: suppress stuck if loop fired with
      content_similarity trigger
    - derive_stop_signals for thrash/runaway_cost from final snapshot
    """
    if not snapshots:
        return {d: False for d in DETECTORS}

    cfg = config or VitalsConfig.from_yaml()
    predictions: dict[str, bool] = {d: False for d in DETECTORS}

    loop_fired = False
    stuck_fired = False
    has_content_similarity = False
    preserve_stuck_overlap = False
    best_loop_trigger: str | None = None
    best_loop_confidence = 0.0
    thrash_fired = False
    runaway_fired = False
    runaway_from_burn_rate = False  # Track burn_rate_anomaly from detect_loop
    last_detector_priority: str | None = None  # Final step's priority for tiebreaker

    # Replay through detect_loop step by step (accumulating history)
    for i, snapshot in enumerate(snapshots):
        history = snapshots[:i] if i > 0 else None
        result = detect_loop(snapshot, history, config=cfg, workflow_type=workflow_type)

        # Track best loop trigger for co-occurrence resolution
        if result.loop_detected and result.loop_confidence > best_loop_confidence:
            best_loop_confidence = result.loop_confidence
            best_loop_trigger = result.loop_trigger
        if result.loop_detected and result.loop_trigger == "content_similarity":
            has_content_similarity = True

        # Preserve stuck when it's the priority detector in a co-occurrence
        if (
            result.loop_detected
            and result.stuck_detected
            and result.detector_priority == "stuck"
        ):
            preserve_stuck_overlap = True

        if result.loop_detected:
            loop_fired = True

        # Stuck suppression rules (matching backtest.py)
        if (
            result.stuck_detected
            and result.detector_priority != "confabulation"
            and result.stuck_trigger != "burn_rate_anomaly"
        ):
            stuck_fired = True

        if result.confabulation_detected:
            predictions["confabulation"] = True

        # Track the final step's detector_priority for co-occurrence tiebreaker.
        # The last step has the most evidence and gives the best determination
        # when content_similarity isn't available (e.g. synthetic traces).
        if result.detector_priority:
            last_detector_priority = result.detector_priority

        # Bridge burn_rate_anomaly → runaway_cost: detect_loop computes
        # burn_rate_anomaly as a stuck candidate, then suppresses stuck_detected.
        # The stuck_trigger remains on the result for stop-rule derivation.
        # Raw snapshots lack stuck_trigger, so derive_stop_signals can't find it.
        # Capture it here directly from the detect_loop result.
        if result.stuck_trigger == "burn_rate_anomaly":
            runaway_from_burn_rate = True

    # Derive stop signals from the final snapshot for thrash/runaway_cost
    final = snapshots[-1]
    stop = derive_stop_signals(
        final,
        step_count=len(snapshots),
        thrash_error_threshold=1,
    )
    if stop.thrash_detected:
        thrash_fired = True
    # Bridge burn_rate to runaway only when stuck isn't the root cause.
    # If stuck fired at any step, the token growth is a symptom of being stuck,
    # not an independent runaway cost issue.
    if stop.runaway_cost_detected or (runaway_from_burn_rate and not stuck_fired):
        runaway_fired = True

    # Co-occurrence resolution (matching backtest.py av32-m02):
    # Only suppress stuck when loop has strong content-based evidence
    # AND stuck was not the priority detector at any step.
    if stuck_fired and loop_fired and not (thrash_fired or runaway_fired):
        if (
            (has_content_similarity or best_loop_trigger == "content_similarity")
            and not preserve_stuck_overlap
        ):
            stuck_fired = False

    predictions["loop"] = loop_fired
    predictions["stuck"] = stuck_fired
    predictions["thrash"] = thrash_fired
    predictions["runaway_cost"] = runaway_fired

    return predictions


def run_evaluation(
    corpus_version: str = "v1",
    detectors: tuple[str, ...] | None = None,
    min_confidence: float = 0.8,
    config: VitalsConfig | None = None,
) -> EvaluationResult:
    """Run full evaluation pipeline against a corpus version."""
    target_detectors = detectors or DETECTORS
    manifest = load_manifest(corpus_version)

    trace_results: list[TraceResult] = []

    for entry in manifest:
        # Filter by confidence
        meta_confidence = entry.get("metadata", {}).get("confidence", 0.0)
        if meta_confidence < min_confidence:
            continue

        trace_id = entry["trace_id"]
        labels = entry["labels"]
        trace_path = entry["path"]

        snapshots = load_trace(corpus_version, trace_path)

        # Resolve workflow type from trace naming (bc=build, rc=research)
        mission_id = snapshots[0].mission_id if snapshots else None
        wf_type = resolve_workflow_type(trace_id, mission_id)

        predictions = replay_trace(snapshots, config, workflow_type=wf_type)

        trace_results.append(TraceResult(
            trace_id=trace_id,
            labels=labels,
            predictions=predictions,
            confidence=meta_confidence,
        ))

    # Compute metrics per detector
    detector_metrics = compute_metrics(trace_results, target_detectors)

    # Check gates
    gate_results = check_all_gates(detector_metrics)

    return EvaluationResult(
        corpus_version=corpus_version,
        trace_count=len(trace_results),
        trace_results=trace_results,
        detector_metrics=detector_metrics,
        gate_results=gate_results,
        detectors_evaluated=target_detectors,
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
    args = parser.parse_args()

    detectors_arg: tuple[str, ...] | None = None
    if args.detectors != "all":
        detectors_arg = tuple(d.strip() for d in args.detectors.split(","))

    result = run_evaluation(
        corpus_version=args.corpus,
        detectors=detectors_arg,
        min_confidence=args.min_confidence,
    )

    report = report_results(result)
    print(report)

    # Exit with non-zero if any evaluated detector failed gate
    any_failed = any(
        not g.get("passed", False)
        for g in result.gate_results.values()
    )
    sys.exit(1 if any_failed and result.trace_count > 0 else 0)


if __name__ == "__main__":
    main()
