"""Expand corpus v1 with delayed-onset confabulation traces.

This script preserves the mixed-source v1 manifest by replacing only the
delayed-onset synthetic confab slice instead of rewriting the entire corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_vitals.schema import VitalsSnapshot

from generators.confabulation import CONFAB_DELAYED_PATTERNS, ConfabGenerator, ConfabPattern

CORPUS_V1 = Path(__file__).resolve().parent.parent / "corpus" / "v1"
POSITIVE_DIR = CORPUS_V1 / "traces" / "confabulation" / "positive"
NEGATIVE_DIR = CORPUS_V1 / "traces" / "confabulation" / "negative"
TRACE_PREFIXES = ("confab-syn-delayed-", "confab-syn-delayed-neg-")


def write_trace(
    *,
    corpus_root: Path,
    traces_dir: Path,
    snapshots: list[VitalsSnapshot],
    trace_id: str,
) -> str:
    """Write a trace JSON document and return its manifest-relative path."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    data = [snapshot.model_dump(mode="json") for snapshot in snapshots]
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path.relative_to(corpus_root))


def cleanup_existing_delayed_traces(*, corpus_root: Path = CORPUS_V1) -> None:
    """Remove previously generated delayed-onset trace files for idempotent re-runs."""
    for traces_dir in (
        corpus_root / "traces" / "confabulation" / "positive",
        corpus_root / "traces" / "confabulation" / "negative",
    ):
        for prefix in TRACE_PREFIXES:
            for path in traces_dir.glob(f"{prefix}*.json"):
                path.unlink(missing_ok=True)


def build_delayed_confab_entries(*, corpus_root: Path = CORPUS_V1) -> list[dict]:
    """Generate delayed-onset confab positives and recovery negatives."""
    generator = ConfabGenerator()
    entries: list[dict] = []
    cleanup_existing_delayed_traces(corpus_root=corpus_root)

    positive_configs: dict[ConfabPattern, list[tuple[int, int, float, float]]] = {
        "delayed_sharp": [
            (10, 2, 0.18, 0.22),
            (10, 3, 0.16, 0.35),
            (12, 2, 0.2, 0.22),
            (12, 3, 0.18, 0.35),
            (12, 4, 0.16, 0.42),
            (14, 3, 0.2, 0.22),
            (14, 4, 0.18, 0.35),
            (14, 5, 0.16, 0.42),
            (15, 3, 0.2, 0.35),
            (15, 4, 0.18, 0.42),
            (15, 5, 0.16, 0.35),
            (16, 5, 0.18, 0.42),
        ],
        "delayed_gradual": [
            (10, 2, 0.2, 0.2),
            (10, 3, 0.18, 0.28),
            (12, 2, 0.22, 0.2),
            (12, 3, 0.2, 0.28),
            (12, 4, 0.18, 0.35),
            (14, 3, 0.22, 0.2),
            (14, 4, 0.2, 0.28),
            (14, 5, 0.18, 0.35),
            (15, 3, 0.22, 0.28),
            (15, 4, 0.2, 0.35),
            (16, 4, 0.18, 0.35),
            (16, 5, 0.16, 0.42),
        ],
        "delayed_oscillating": [
            (10, 2, 0.2, 0.24),
            (10, 3, 0.18, 0.32),
            (12, 2, 0.22, 0.24),
            (12, 3, 0.2, 0.32),
            (12, 4, 0.18, 0.4),
            (14, 3, 0.22, 0.24),
            (14, 4, 0.2, 0.32),
            (14, 5, 0.18, 0.4),
            (15, 3, 0.22, 0.32),
            (15, 4, 0.2, 0.4),
            (16, 4, 0.18, 0.4),
            (16, 5, 0.16, 0.32),
        ],
    }

    positive_seq = 1
    for pattern in CONFAB_DELAYED_PATTERNS:
        for total_steps, onset_step, ratio, confidence_inflation in positive_configs[pattern]:
            trace_id = f"confab-syn-delayed-{positive_seq:03d}"
            snapshots, metadata = generator.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                onset_step=onset_step,
                source_finding_ratio=ratio,
                confidence_inflation=confidence_inflation,
                pattern=pattern,
                positive=True,
            )
            rel_path = write_trace(
                corpus_root=corpus_root,
                traces_dir=corpus_root / "traces" / "confabulation" / "positive",
                snapshots=snapshots,
                trace_id=trace_id,
            )
            entries.append(metadata.as_manifest_entry(rel_path))
            positive_seq += 1

    negative_configs: dict[ConfabPattern, list[tuple[int, int, int, float]]] = {
        "delayed_sharp": [
            (10, 3, 5, 0.2),
            (12, 4, 6, 0.18),
            (14, 5, 7, 0.16),
        ],
        "delayed_gradual": [
            (10, 3, 6, 0.2),
            (12, 4, 7, 0.18),
            (14, 5, 8, 0.16),
        ],
        "delayed_oscillating": [
            (10, 3, 6, 0.2),
            (12, 4, 7, 0.18),
            (14, 5, 8, 0.16),
        ],
    }

    negative_seq = 1
    for pattern in CONFAB_DELAYED_PATTERNS:
        for total_steps, onset_step, recovery_step, ratio in negative_configs[pattern]:
            trace_id = f"confab-syn-delayed-neg-{negative_seq:03d}"
            snapshots, metadata = generator.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                onset_step=onset_step,
                recovery_step=recovery_step,
                source_finding_ratio=ratio,
                confidence_inflation=0.18,
                pattern=pattern,
                positive=False,
            )
            rel_path = write_trace(
                corpus_root=corpus_root,
                traces_dir=corpus_root / "traces" / "confabulation" / "negative",
                snapshots=snapshots,
                trace_id=trace_id,
            )
            entries.append(metadata.as_manifest_entry(rel_path))
            negative_seq += 1

    return entries


def merge_manifest(
    *,
    entries: list[dict],
    manifest_path: Path = CORPUS_V1 / "manifest.json",
) -> list[dict]:
    """Merge delayed-onset entries into the v1 manifest."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []

    manifest = [
        entry
        for entry in manifest
        if not entry["trace_id"].startswith(TRACE_PREFIXES)
    ]
    manifest.extend(entries)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    entries = build_delayed_confab_entries()
    manifest = merge_manifest(entries=entries)
    positives = sum(1 for entry in entries if entry["labels"]["confabulation"])
    negatives = len(entries) - positives
    print(f"Generated delayed confab entries: pos={positives} neg={negatives} total={len(entries)}")
    print(f"Updated manifest: {CORPUS_V1 / 'manifest.json'}")
    print(f"Total manifest entries: {len(manifest)}")


if __name__ == "__main__":
    main()
