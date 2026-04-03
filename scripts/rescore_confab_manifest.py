#!/usr/bin/env python3
"""Re-score confab manifest entries that were incorrectly flagged as cross-fire.

The original cross_validate_trace flagged ALL stuck triggers on confab traces
as cross-fire, but confab traces legitimately co-trigger stuck (coverage
stagnates when sources are fabricated). The evaluator handles this via
detector_priority="confabulation" suppression.

This script re-runs validation with corrected logic and updates confidence.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.stop_rule import derive_stop_signals
from agent_vitals.schema import VitalsSnapshot

CORPUS_DIR = PROJECT_ROOT / "corpus" / "v1"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


def revalidate_confab_trace(snapshots):
    """Re-run cross-validation with corrected logic for confab traces."""
    cfg = VitalsConfig(min_evidence_steps=3, model_size_class="auto")
    stuck_cross = False
    confab_detected = False

    for i in range(2, len(snapshots)):
        current = snapshots[i]
        history = snapshots[:i]
        result = detect_loop(current, history, config=cfg, workflow_type="research")

        if result.confabulation_detected:
            confab_detected = True

        # Only flag cross-fire when stuck fires AND confab doesn't have priority
        if (
            result.stuck_detected
            and result.stuck_trigger != "burn_rate_anomaly"
            and result.detector_priority != "confabulation"
        ):
            stuck_cross = True

    return {"stuck_cross_fire": stuck_cross, "confab_detected": confab_detected}


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    updated = 0

    for entry in manifest:
        # Only re-score confab traces that were flagged as cross-fire
        if not entry.get("labels", {}).get("confabulation"):
            continue
        cv = entry.get("metadata", {}).get("cross_validation", {})
        if not cv.get("stuck_cross_fire"):
            continue

        # Load trace
        trace_path = CORPUS_DIR / entry["path"]
        if not trace_path.exists():
            print(f"  SKIP {entry['trace_id']}: file not found")
            continue

        snapshots_data = json.loads(trace_path.read_text())
        snapshots = [VitalsSnapshot.model_validate(s) for s in snapshots_data]

        # Re-validate
        result = revalidate_confab_trace(snapshots)

        old_conf = entry["metadata"]["confidence"]
        if not result["stuck_cross_fire"]:
            # Restore original confidence (before cross-fire downgrade)
            new_conf = 0.9 if result["confab_detected"] else 0.8
            entry["metadata"]["confidence"] = new_conf
            entry["metadata"]["cross_validation"]["stuck_cross_fire"] = False
            updated += 1
            print(f"  FIXED {entry['trace_id']}: {old_conf} -> {new_conf}")
        else:
            print(f"  KEEP {entry['trace_id']}: still cross-fires (conf={old_conf})")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nUpdated {updated} entries")


if __name__ == "__main__":
    main()
