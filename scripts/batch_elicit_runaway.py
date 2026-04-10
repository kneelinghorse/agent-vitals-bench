#!/usr/bin/env python3
"""Batch runaway cost elicitation with cross-detector validation and corpus merge.

Runs elicitation against specified providers, validates each trace for
cross-detector issues, and merges passing traces into corpus/v1/.

Usage:
    python scripts/batch_elicit_runaway.py --providers qwen3.5-27b qwen3.5-9b-a \
        --positive-count 40 --negative-count 20 --steps 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.stop_rule import derive_stop_signals

from elicitation.elicit_runaway_cost import elicit_runaway_cost
from elicitation.providers import get_provider

CORPUS_DIR = PROJECT_ROOT / "corpus" / "v1"
TRACES_DIR = CORPUS_DIR / "traces" / "runaway_cost" / "elicited"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


def cross_validate_trace(snapshots, *, config=None):
    """Run cross-detector validation on a trace.

    Returns dict with validation results:
      - stuck_cross_fire: True if stuck fires on a runaway trace
      - runaway_detected: True if runaway_cost is properly detected
      - triggers: list of (stuck_trigger, detector_priority) per step
    """
    cfg = config or VitalsConfig(min_evidence_steps=3, model_size_class="auto")
    triggers = []
    stuck_fired = False
    runaway_fired = False

    for i in range(2, len(snapshots)):
        current = snapshots[i]
        history = snapshots[:i]
        result = detect_loop(current, history, config=cfg, workflow_type="research")

        snapshot_dict = current.model_dump()
        snapshot_dict.update(result.as_snapshot_update())
        signals = derive_stop_signals(snapshot_dict, step_count=i + 1)

        # agent-vitals v1.14.2 replaced the implicit `stuck_trigger ==
        # "burn_rate_anomaly"` sentinel with an explicit
        # `runaway_cost_detected` field on LoopDetectionResult. The check
        # below means "stuck genuinely fired and is not the burn-rate
        # runaway side channel" — under v1.14.2 we read the explicit field
        # directly instead of relying on the legacy sentinel string.
        if (
            result.stuck_detected
            and not result.runaway_cost_detected
            and result.detector_priority != "runaway_cost"
        ):
            stuck_fired = True
        if signals.runaway_cost_detected:
            runaway_fired = True

        triggers.append({
            "step": i,
            "stuck_detected": result.stuck_detected,
            "stuck_trigger": result.stuck_trigger,
            "runaway_cost": signals.runaway_cost_detected,
            "detector_priority": result.detector_priority,
        })

    return {
        "stuck_cross_fire": stuck_fired,
        "runaway_detected": runaway_fired,
        "triggers": triggers,
    }


def save_trace(run, validation, output_dir):
    """Save a trace to the corpus directory."""
    subdir = "positive" if run.positive else "negative"
    trace_dir = output_dir / subdir
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_path = trace_dir / f"{run.trace_id}.json"
    snapshots_data = [s.model_dump(mode="json") for s in run.snapshots]
    trace_path.write_text(json.dumps(snapshots_data, indent=2))

    return trace_path


def build_manifest_entry(run, validation, trace_path, corpus_dir):
    """Build a manifest entry for a trace."""
    rel_path = str(trace_path.relative_to(corpus_dir))

    # Adjust confidence based on cross-validation
    confidence = run.metadata.confidence
    if validation["stuck_cross_fire"]:
        confidence = min(confidence, 0.5)  # Downgrade on cross-fire

    return {
        "trace_id": run.trace_id,
        "path": rel_path,
        "tier": "elicited",
        "labels": run.metadata.labels,
        "metadata": {
            "generator": run.metadata.generator,
            "params": run.metadata.params,
            "onset_step": run.metadata.onset_step,
            "tier": "elicited",
            "model": run.model,
            "provider": run.provider_name,
            "framework": None,
            "reviewer": None,
            "review_date": None,
            "confidence": confidence,
            "notes": run.metadata.notes,
            "cross_validation": {
                "stuck_cross_fire": validation["stuck_cross_fire"],
                "runaway_detected": validation["runaway_detected"],
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Batch runaway cost elicitation")
    parser.add_argument(
        "--providers", nargs="+", default=["qwen3.5-27b"],
        help="Provider names to elicit from",
    )
    parser.add_argument("--positive-count", type=int, default=40)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    if args.dry_run:
        total = (args.positive_count + args.negative_count) * len(args.providers)
        print(f"Would elicit {total} traces ({args.positive_count}+/{args.negative_count}-) "
              f"from {args.providers} at {args.steps} steps each")
        return

    # Load existing manifest
    manifest = []
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    existing_ids = {e["trace_id"] for e in manifest}

    stats = {
        "total": 0,
        "passed": 0,
        "cross_fire": 0,
        "skipped_duplicate": 0,
        "by_provider": {},
    }

    for provider_name in args.providers:
        print(f"\n{'='*60}")
        print(f"Provider: {provider_name}")
        print(f"{'='*60}")
        provider = get_provider(provider_name)
        provider_stats = {"positive_passed": 0, "negative_passed": 0, "cross_fire": 0}

        # Positive traces
        for i in range(args.positive_count):
            stats["total"] += 1
            print(f"  [{i+1}/{args.positive_count}] Positive elicitation...", end=" ", flush=True)

            try:
                run = elicit_runaway_cost(provider, positive=True, total_steps=args.steps)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if run.trace_id in existing_ids:
                stats["skipped_duplicate"] += 1
                print(f"SKIP (duplicate)")
                continue

            validation = cross_validate_trace(run.snapshots)
            trace_path = save_trace(run, validation, TRACES_DIR)
            entry = build_manifest_entry(run, validation, trace_path, CORPUS_DIR)
            manifest.append(entry)
            existing_ids.add(run.trace_id)

            if validation["stuck_cross_fire"]:
                stats["cross_fire"] += 1
                provider_stats["cross_fire"] += 1
                print(f"CROSS-FIRE (conf={entry['metadata']['confidence']})")
            else:
                stats["passed"] += 1
                provider_stats["positive_passed"] += 1
                conf = entry["metadata"]["confidence"]
                ratio = run.metadata.params.get("peak_ratio", "?")
                print(f"OK conf={conf} ratio={ratio}x")

        # Negative traces
        for i in range(args.negative_count):
            stats["total"] += 1
            print(f"  [{i+1}/{args.negative_count}] Negative control...", end=" ", flush=True)

            try:
                run = elicit_runaway_cost(provider, positive=False, total_steps=args.steps)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if run.trace_id in existing_ids:
                stats["skipped_duplicate"] += 1
                print(f"SKIP (duplicate)")
                continue

            validation = cross_validate_trace(run.snapshots)
            trace_path = save_trace(run, validation, TRACES_DIR)
            entry = build_manifest_entry(run, validation, trace_path, CORPUS_DIR)
            manifest.append(entry)
            existing_ids.add(run.trace_id)

            stats["passed"] += 1
            provider_stats["negative_passed"] += 1
            conf = entry["metadata"]["confidence"]
            print(f"OK conf={conf}")

        stats["by_provider"][provider_name] = provider_stats

    # Save updated manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total traces: {stats['total']}")
    print(f"Passed validation: {stats['passed']}")
    print(f"Cross-fire (downgraded): {stats['cross_fire']}")
    print(f"Skipped duplicates: {stats['skipped_duplicate']}")
    for prov, pstats in stats["by_provider"].items():
        print(f"  {prov}: {pstats['positive_passed']}+/{pstats['negative_passed']}- "
              f"({pstats['cross_fire']} cross-fire)")

    # Count total runaway traces in corpus at >= 0.8 confidence
    high_conf = sum(
        1 for e in manifest
        if e.get("labels", {}).get("runaway_cost")
        and e.get("metadata", {}).get("confidence", 0) >= 0.8
    )
    print(f"\nTotal runaway_cost traces at conf >= 0.8: {high_conf}")


if __name__ == "__main__":
    main()
