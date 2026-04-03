"""Migrate AV-34 corpus into agent-vitals-bench legacy corpus.

Applies the triage rules from corpus_schema.md:
- Accept loop/stuck traces with confidence 0.7 (legacy, reviewed)
- Demote confab/thrash/runaway_cost labels (re-review pending)
- Exclude traces with < min_evidence_steps (3) steps
- Organize into detector/polarity directories
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

AV34_ROOT = Path(__file__).resolve().parent.parent.parent / "agent-vitals" / "checkpoints" / "vitals_corpus" / "av34_reviewed_merged"
CORPUS_LEGACY = Path(__file__).resolve().parent.parent / "corpus" / "legacy"

MIN_STEPS = 3


def extract_model_from_name(trace_name: str) -> str | None:
    """Extract model identifier from trace filename."""
    parts = trace_name.split("-")
    if len(parts) >= 2 and parts[0].startswith("av3"):
        return parts[1]
    return parts[0] if parts else None


def count_steps(trace_path: Path) -> int:
    """Count non-empty lines (snapshots) in a JSONL trace."""
    return sum(1 for line in trace_path.read_text().strip().splitlines() if line.strip())


def main() -> None:
    labels_path = AV34_ROOT / "labels.json"
    traces_dir = AV34_ROOT / "traces"

    labels = json.loads(labels_path.read_text())
    print(f"AV-34 corpus: {len(labels)} traces")

    # Ensure output directories exist
    for detector in ("loop", "stuck"):
        for polarity in ("positive", "negative"):
            (CORPUS_LEGACY / "traces" / detector / polarity).mkdir(parents=True, exist_ok=True)
    (CORPUS_LEGACY / "demoted").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    stats = {
        "total": 0,
        "excluded_short": 0,
        "migrated": 0,
        "demoted_confab": 0,
        "demoted_thrash": 0,
        "demoted_runaway": 0,
        "loop_pos": 0,
        "loop_neg": 0,
        "stuck_pos": 0,
        "stuck_neg": 0,
    }

    for trace_name, trace_labels in sorted(labels.items()):
        stats["total"] += 1
        trace_file = traces_dir / f"{trace_name}.jsonl"

        if not trace_file.exists():
            print(f"  SKIP (missing): {trace_name}")
            continue

        step_count = count_steps(trace_file)
        if step_count < MIN_STEPS:
            stats["excluded_short"] += 1
            continue

        # Determine labels — only loop and stuck are trusted for legacy
        loop_positive = len(trace_labels.get("loop_at", [])) > 0
        stuck_positive = len(trace_labels.get("stuck_at", [])) > 0
        confab_positive = len(trace_labels.get("confabulation_at", [])) > 0
        thrash_positive = len(trace_labels.get("thrash_at", [])) > 0
        runaway_positive = len(trace_labels.get("runaway_cost_at", [])) > 0

        # Track demoted labels
        if confab_positive:
            stats["demoted_confab"] += 1
        if thrash_positive:
            stats["demoted_thrash"] += 1
        if runaway_positive:
            stats["demoted_runaway"] += 1

        # Determine primary detector for filing
        if loop_positive:
            primary = "loop"
            polarity = "positive"
            stats["loop_pos"] += 1
        elif stuck_positive:
            primary = "stuck"
            polarity = "positive"
            stats["stuck_pos"] += 1
        else:
            # Negative for both — file under loop/negative (arbitrary, manifest has truth)
            primary = "loop"
            polarity = "negative"
            stats["loop_neg"] += 1

        # Count stuck negatives (traces that are not stuck-positive)
        if not stuck_positive:
            stats["stuck_neg"] += 1

        # Copy trace file
        dest_dir = CORPUS_LEGACY / "traces" / primary / polarity
        dest_file = dest_dir / f"{trace_name}.jsonl"
        shutil.copy2(trace_file, dest_file)

        # Build manifest entry
        model = extract_model_from_name(trace_name)
        onset_step = None
        if loop_positive:
            onsets = trace_labels.get("loop_at", [])
            onset_step = min(onsets) if onsets else None
        elif stuck_positive:
            onsets = trace_labels.get("stuck_at", [])
            onset_step = min(onsets) if onsets else None

        rel_path = f"traces/{primary}/{polarity}/{trace_name}.jsonl"

        entry = {
            "trace_id": trace_name,
            "path": rel_path,
            "tier": "legacy",
            "labels": {
                "loop": loop_positive,
                "stuck": stuck_positive,
                # Demote confab/thrash/runaway — set False regardless
                "confabulation": False,
                "thrash": False,
                "runaway_cost": False,
            },
            "metadata": {
                "generator": None,
                "params": None,
                "onset_step": onset_step,
                "tier": "legacy",
                "model": model,
                "provider": "openrouter" if model and model.startswith("or_") else model,
                "framework": "langchain",
                "reviewer": "av34-review-pipeline",
                "review_date": "2026-03-29",
                "confidence": 0.7,
                "notes": f"AV-34 legacy. steps={step_count}",
                "original_labels": {
                    "confabulation": confab_positive,
                    "thrash": thrash_positive,
                    "runaway_cost": runaway_positive,
                },
            },
        }
        manifest.append(entry)
        stats["migrated"] += 1

    # Write manifest
    manifest_path = CORPUS_LEGACY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nMigration complete:")
    print(f"  Total in AV-34:     {stats['total']}")
    print(f"  Excluded (< {MIN_STEPS} steps): {stats['excluded_short']}")
    print(f"  Migrated:           {stats['migrated']}")
    print(f"  Loop positive:      {stats['loop_pos']}")
    print(f"  Loop negative:      {stats['loop_neg']}")
    print(f"  Stuck positive:     {stats['stuck_pos']}")
    print(f"  Stuck negative:     {stats['stuck_neg']}")
    print(f"  Demoted confab:     {stats['demoted_confab']}")
    print(f"  Demoted thrash:     {stats['demoted_thrash']}")
    print(f"  Demoted runaway:    {stats['demoted_runaway']}")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
