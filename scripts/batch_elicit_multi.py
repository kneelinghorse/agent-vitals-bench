#!/usr/bin/env python3
"""Multi-provider, multi-detector batch elicitation.

Runs confab, thrash, and runaway elicitation across API providers,
validates traces, and merges into corpus/v1/.

Usage:
    python scripts/batch_elicit_multi.py \
        --detectors confabulation thrash runaway_cost \
        --providers deepseek minimax openrouter-gemini \
        --positive-count 10 --negative-count 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.stop_rule import derive_stop_signals

from elicitation.providers import get_provider

CORPUS_DIR = PROJECT_ROOT / "corpus" / "v1"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

# Map detector names to elicitor functions and corpus subdirs
DETECTOR_CONFIG = {
    "confabulation": {
        "import": "elicitation.elicit_confabulation",
        "function": "elicit_confabulation",
        "corpus_subdir": "confabulation",
        "default_steps": 6,
    },
    "thrash": {
        "import": "elicitation.elicit_thrash",
        "function": "elicit_thrash",
        "corpus_subdir": "thrash",
        "default_steps": 8,
    },
    "runaway_cost": {
        "import": "elicitation.elicit_runaway_cost",
        "function": "elicit_runaway_cost",
        "corpus_subdir": "runaway_cost",
        "default_steps": 8,
    },
}


def get_elicitor(detector: str):
    """Dynamically import and return the elicitor function."""
    cfg = DETECTOR_CONFIG[detector]
    mod = __import__(cfg["import"], fromlist=[cfg["function"]])
    return getattr(mod, cfg["function"])


def cross_validate_trace(snapshots, detector: str):
    """Run cross-detector validation. Returns dict with results."""
    cfg = VitalsConfig(min_evidence_steps=3, model_size_class="auto")
    stuck_cross = False
    target_detected = False

    for i in range(2, len(snapshots)):
        current = snapshots[i]
        history = snapshots[:i]
        result = detect_loop(current, history, config=cfg, workflow_type="research")
        snapshot_dict = current.model_dump()
        snapshot_dict.update(result.as_snapshot_update())
        signals = derive_stop_signals(snapshot_dict, step_count=i + 1)

        # Check if target detector fires
        if detector == "confabulation" and result.confabulation_detected:
            target_detected = True
        elif detector == "thrash" and signals.thrash_detected:
            target_detected = True
        elif detector == "runaway_cost" and signals.runaway_cost_detected:
            target_detected = True

        # Check for unwanted cross-fire.
        # Confab traces legitimately co-trigger stuck (coverage stagnates when
        # sources are fabricated). The evaluator handles this via
        # detector_priority="confabulation" suppression (runner.py:150-151).
        # Only flag as cross-fire when stuck fires WITHOUT the expected
        # detector priority for the target detector.
        if (
            detector != "stuck"
            and result.stuck_detected
            and result.stuck_trigger != "burn_rate_anomaly"
            and result.detector_priority != detector
        ):
            stuck_cross = True

    return {
        "target_detected": target_detected,
        "stuck_cross_fire": stuck_cross,
    }


def save_trace(run, detector: str, output_dir: Path):
    """Save trace to corpus."""
    subdir = "positive" if run.positive else "negative"
    trace_dir = output_dir / DETECTOR_CONFIG[detector]["corpus_subdir"] / "elicited" / subdir
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{run.trace_id}.json"
    data = [s.model_dump(mode="json") for s in run.snapshots]
    trace_path.write_text(json.dumps(data, indent=2))
    return trace_path


def build_manifest_entry(run, validation, trace_path, corpus_dir):
    """Build manifest entry."""
    confidence = run.metadata.confidence
    if validation["stuck_cross_fire"]:
        confidence = min(confidence, 0.5)

    return {
        "trace_id": run.trace_id,
        "path": str(trace_path.relative_to(corpus_dir)),
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
                "target_detected": validation["target_detected"],
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-provider batch elicitation")
    parser.add_argument(
        "--detectors", nargs="+",
        default=["confabulation", "thrash", "runaway_cost"],
        choices=list(DETECTOR_CONFIG),
    )
    parser.add_argument(
        "--providers", nargs="+",
        default=["deepseek", "minimax", "openrouter-gemini"],
    )
    parser.add_argument("--positive-count", type=int, default=10)
    parser.add_argument("--negative-count", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        total = (args.positive_count + args.negative_count) * len(args.providers) * len(args.detectors)
        print(f"Would elicit {total} traces across {args.detectors} x {args.providers}")
        return

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    existing_ids = {e["trace_id"] for e in manifest}
    traces_dir = CORPUS_DIR / "traces"

    stats = {"total": 0, "passed": 0, "cross_fire": 0, "errors": 0}

    for detector in args.detectors:
        elicit_fn = get_elicitor(detector)
        default_steps = DETECTOR_CONFIG[detector]["default_steps"]

        for provider_name in args.providers:
            print(f"\n{'='*60}")
            print(f"{detector} × {provider_name}")
            print(f"{'='*60}")

            try:
                provider = get_provider(provider_name)
            except Exception as e:
                print(f"  SKIP provider: {e}")
                continue

            for polarity, count in [("positive", args.positive_count), ("negative", args.negative_count)]:
                for i in range(count):
                    stats["total"] += 1
                    is_positive = polarity == "positive"
                    label = "+" if is_positive else "-"
                    print(f"  [{i+1}/{count}] {polarity}...", end=" ", flush=True)

                    try:
                        kwargs = {"positive": is_positive, "total_steps": default_steps}
                        if detector == "confabulation":
                            kwargs["verify_api_key"] = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
                        run = elicit_fn(provider, **kwargs)
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"ERROR: {e}")
                        continue

                    if run.trace_id in existing_ids:
                        print("SKIP (dup)")
                        continue

                    validation = cross_validate_trace(run.snapshots, detector)
                    trace_path = save_trace(run, detector, traces_dir)
                    entry = build_manifest_entry(run, validation, trace_path, CORPUS_DIR)
                    manifest.append(entry)
                    existing_ids.add(run.trace_id)

                    conf = entry["metadata"]["confidence"]
                    if validation["stuck_cross_fire"]:
                        stats["cross_fire"] += 1
                        print(f"CROSS-FIRE conf={conf}")
                    else:
                        stats["passed"] += 1
                        print(f"OK conf={conf}")

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}  Passed: {stats['passed']}  "
          f"Cross-fire: {stats['cross_fire']}  Errors: {stats['errors']}")

    # Provider coverage per detector
    from collections import defaultdict
    prov_det = defaultdict(set)
    for e in manifest:
        if e.get("tier") != "elicited":
            continue
        prov = e.get("metadata", {}).get("params", {}).get("provider") or e.get("metadata", {}).get("provider")
        for label, val in e.get("labels", {}).items():
            if val and prov:
                prov_det[label].add(prov)
    print(f"\nProvider coverage:")
    for det in ["confabulation", "thrash", "runaway_cost"]:
        provs = prov_det.get(det, set())
        print(f"  {det}: {len(provs)} providers — {sorted(provs)}")


if __name__ == "__main__":
    main()
