#!/usr/bin/env python3
"""Step-level detector co-occurrence analysis for the 32 remaining runaway_cost FPs.

Answers agent-vitals S11-M01 intel request:
1. Per-step detector firing history for each of the 32 FP traces
2. Same-step stuck+runaway co-occurrence counts
3. Mixed-label trace analysis (traces labeled both stuck=true and runaway_cost=true)

Output:
  reports/runaway-step-cooccurrence.json
  reports/runaway-step-cooccurrence.md

Usage:
    python scripts/runaway_step_cooccurrence.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_vitals import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.stop_rule import derive_stop_signals
from agent_vitals.schema import VitalsSnapshot

from evaluator.runner import load_manifest, load_trace, replay_trace, DETECTORS

DETECTOR_FIELDS = {
    "loop": "loop_detected",
    "stuck": "stuck_detected",
    "confabulation": "confabulation_detected",
    "thrash": "thrash_detected",
    "runaway_cost": "runaway_cost_detected",
}

CONFIDENCE_FIELDS = {
    "loop": "loop_confidence",
    "stuck": "stuck_confidence",
    "confabulation": "confabulation_confidence",
}


def step_level_replay(
    snapshots: list[VitalsSnapshot],
    config: VitalsConfig,
    workflow_type: str = "unknown",
) -> list[dict]:
    """Replay trace step-by-step, capturing per-step detector firings."""
    steps = []
    history: list[VitalsSnapshot] = []

    for i, snapshot in enumerate(snapshots):
        detection = detect_loop(
            snapshot,
            history,
            config=config,
            workflow_type=workflow_type,
        )

        step_data = {
            "step": i,
            "loop_index": snapshot.loop_index,
            "coverage_score": float(snapshot.signals.coverage_score),
            "findings_count": int(snapshot.signals.findings_count),
            "total_tokens": int(snapshot.signals.total_tokens),
            "detectors_fired": {},
        }

        for det, field in DETECTOR_FIELDS.items():
            fired = getattr(detection, field, False)
            if fired:
                confidence = getattr(detection, CONFIDENCE_FIELDS.get(det, ""), None)
                step_data["detectors_fired"][det] = {
                    "fired": True,
                    "confidence": round(float(confidence), 4) if confidence is not None else None,
                }

        # Additional context for stuck
        if detection.stuck_detected:
            step_data["detectors_fired"].setdefault("stuck", {})
            step_data["detectors_fired"]["stuck"]["trigger"] = detection.stuck_trigger

        steps.append(step_data)
        history.append(snapshot)

    return steps


def main() -> None:
    print("Step-Level Detector Co-occurrence Analysis")
    print("=" * 50)

    cfg = VitalsConfig.from_yaml()
    manifest = load_manifest("v1")

    # Identify the 32 remaining FPs (post-v1.16.0 multiplier 3.0)
    runaway_entries = [
        e for e in manifest
        if e["labels"].get("runaway_cost") is not None
        and e.get("metadata", {}).get("confidence", 0.0) >= 0.8
    ]

    print(f"Total runaway_cost traces: {len(runaway_entries)}")

    fp_entries = []
    tp_entries = []
    for entry in runaway_entries:
        snaps = load_trace("v1", entry["path"])
        preds = replay_trace(snaps, cfg, runtime_mode="default")
        label = entry["labels"]["runaway_cost"]
        predicted = preds["runaway_cost"]
        if predicted and not label:
            fp_entries.append(entry)
        elif predicted and label:
            tp_entries.append(entry)

    print(f"FPs (post-v1.16.0): {len(fp_entries)}")
    print(f"TPs: {len(tp_entries)}")

    # Identify mixed-label traces (both stuck=true and runaway_cost=true)
    mixed_entries = [
        e for e in manifest
        if e["labels"].get("stuck") is True
        and e["labels"].get("runaway_cost") is True
        and e.get("metadata", {}).get("confidence", 0.0) >= 0.8
    ]
    print(f"Mixed-label traces (stuck+runaway both true): {len(mixed_entries)}")

    # Step-level analysis for FPs
    print("\nAnalyzing FP step-level patterns...")
    fp_analyses = []
    same_step_count = 0
    temporal_sep_count = 0

    for entry in fp_entries:
        tid = entry["trace_id"]
        snaps = load_trace("v1", entry["path"])
        steps = step_level_replay(snaps, cfg)

        # Find co-occurrence
        stuck_steps = [s["step"] for s in steps if "stuck" in s["detectors_fired"]]
        runaway_steps = [s["step"] for s in steps if "runaway_cost" in s["detectors_fired"]]
        overlap_steps = set(stuck_steps) & set(runaway_steps)

        has_same_step = len(overlap_steps) > 0
        if has_same_step:
            same_step_count += 1
        elif stuck_steps and runaway_steps:
            temporal_sep_count += 1

        fp_analyses.append({
            "trace_id": tid,
            "labels": entry["labels"],
            "total_steps": len(steps),
            "stuck_fires_at": stuck_steps,
            "runaway_fires_at": runaway_steps,
            "same_step_overlap": sorted(overlap_steps),
            "has_same_step_cooccurrence": has_same_step,
            "temporal_gap": (
                min(abs(r - s) for r in runaway_steps for s in stuck_steps)
                if stuck_steps and runaway_steps else None
            ),
            "steps": steps,
        })

    # Step-level analysis for mixed-label traces
    print("Analyzing mixed-label traces...")
    mixed_analyses = []
    for entry in mixed_entries:
        tid = entry["trace_id"]
        snaps = load_trace("v1", entry["path"])
        steps = step_level_replay(snaps, cfg)

        stuck_steps = [s["step"] for s in steps if "stuck" in s["detectors_fired"]]
        runaway_steps = [s["step"] for s in steps if "runaway_cost" in s["detectors_fired"]]
        overlap_steps = set(stuck_steps) & set(runaway_steps)

        mixed_analyses.append({
            "trace_id": tid,
            "labels": entry["labels"],
            "total_steps": len(steps),
            "stuck_fires_at": stuck_steps,
            "runaway_fires_at": runaway_steps,
            "same_step_overlap": sorted(overlap_steps),
            "temporal_gap": (
                min(abs(r - s) for r in runaway_steps for s in stuck_steps)
                if stuck_steps and runaway_steps else None
            ),
            "steps": steps,
        })

    # Compute summary stats
    fp_with_stuck = [a for a in fp_analyses if a["stuck_fires_at"]]
    fp_stuck_only = [a for a in fp_analyses if a["stuck_fires_at"] and not a["has_same_step_cooccurrence"]]

    # Build reports
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    json_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": "v1",
        "runtime_mode": "default",
        "agent_vitals_version": "1.16.0",
        "summary": {
            "total_fps": len(fp_entries),
            "fps_with_stuck_cofire": len(fp_with_stuck),
            "fps_with_same_step_cooccurrence": same_step_count,
            "fps_with_temporal_separation": temporal_sep_count,
            "fps_without_stuck": len(fp_entries) - len(fp_with_stuck),
            "mixed_label_traces": len(mixed_entries),
        },
        "fp_traces": fp_analyses,
        "mixed_label_traces": mixed_analyses,
    }

    lines = [
        "# Step-Level Detector Co-occurrence Analysis",
        "",
        f"**Corpus:** v1 | **Mode:** default | **agent-vitals:** v1.16.0 | **Generated:** {now}",
        "",
        "## Summary",
        "",
        f"- **Total remaining FPs:** {len(fp_entries)}",
        f"- FPs where stuck also fires: {len(fp_with_stuck)}",
        f"  - Same-step co-occurrence (stuck+runaway at same step): **{same_step_count}**",
        f"  - Temporally separated (stuck and runaway at different steps): **{temporal_sep_count}**",
        f"- FPs without stuck co-fire: {len(fp_entries) - len(fp_with_stuck)}",
        f"- Mixed-label traces (stuck=true AND runaway=true): {len(mixed_entries)}",
        "",
        "---",
        "",
        "## Co-occurrence Pattern Summary",
        "",
        "| Pattern | Count | Suppression-Safe? |",
        "|---------|-------|-------------------|",
        f"| Same-step co-occurrence | {same_step_count} | YES — window=0 suffices |",
        f"| Temporal separation | {temporal_sep_count} | Depends on gap size |",
        f"| No stuck co-fire | {len(fp_entries) - len(fp_with_stuck)} | N/A — different root cause |",
        "",
        "---",
        "",
        "## FP Trace Details",
        "",
        "| Trace ID | Steps | Stuck fires at | Runaway fires at | Overlap | Gap |",
        "|----------|-------|---------------|-----------------|---------|-----|",
    ]

    for a in fp_analyses:
        stuck_str = ",".join(str(s) for s in a["stuck_fires_at"]) or "—"
        runaway_str = ",".join(str(s) for s in a["runaway_fires_at"]) or "—"
        overlap_str = ",".join(str(s) for s in a["same_step_overlap"]) or "—"
        gap_str = str(a["temporal_gap"]) if a["temporal_gap"] is not None else "—"
        lines.append(
            f"| {a['trace_id']} | {a['total_steps']} "
            f"| {stuck_str} | {runaway_str} | {overlap_str} | {gap_str} |"
        )

    if mixed_analyses:
        lines.extend([
            "",
            "---",
            "",
            "## Mixed-Label Traces (stuck=true AND runaway_cost=true)",
            "",
            "These define the safety boundary — suppression window must NOT clobber these.",
            "",
            "| Trace ID | Steps | Stuck fires at | Runaway fires at | Overlap | Gap |",
            "|----------|-------|---------------|-----------------|---------|-----|",
        ])
        for a in mixed_analyses:
            stuck_str = ",".join(str(s) for s in a["stuck_fires_at"]) or "—"
            runaway_str = ",".join(str(s) for s in a["runaway_fires_at"]) or "—"
            overlap_str = ",".join(str(s) for s in a["same_step_overlap"]) or "—"
            gap_str = str(a["temporal_gap"]) if a["temporal_gap"] is not None else "—"
            lines.append(
                f"| {a['trace_id']} | {a['total_steps']} "
                f"| {stuck_str} | {runaway_str} | {overlap_str} | {gap_str} |"
            )

    report_md = "\n".join(lines) + "\n"

    # Write reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    md_path = reports_dir / "runaway-step-cooccurrence.md"
    md_path.write_text(report_md)
    print(f"\nMarkdown: {md_path}")

    json_path = reports_dir / "runaway-step-cooccurrence.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    print(f"JSON: {json_path}")

    # Print summary
    print(f"\n{'=' * 50}")
    print("CO-OCCURRENCE SUMMARY:")
    print(f"{'=' * 50}")
    print(f"Same-step co-occurrence: {same_step_count}/{len(fp_entries)} FPs")
    print(f"Temporal separation: {temporal_sep_count}/{len(fp_entries)} FPs")
    print(f"No stuck co-fire: {len(fp_entries) - len(fp_with_stuck)}/{len(fp_entries)} FPs")
    print(f"Mixed-label safety traces: {len(mixed_entries)}")

    if mixed_analyses:
        print("\nMixed-label temporal gaps:")
        for a in mixed_analyses:
            gap = a["temporal_gap"]
            print(f"  {a['trace_id']}: gap={gap} steps "
                  f"(stuck@{a['stuck_fires_at']}, runaway@{a['runaway_fires_at']})")


if __name__ == "__main__":
    main()
