"""Run elicitation collection against local llama.cpp instances.

Collects real traces for confabulation, thrash, and runaway_cost detectors.
Merges results into corpus/v1/ with updated manifest.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_vitals.schema import VitalsSnapshot

from elicitation.providers import get_provider, verify_doi, list_providers
from elicitation.elicit_confabulation import elicit_confabulation
from elicitation.elicit_thrash import elicit_thrash
from elicitation.elicit_runaway_cost import elicit_runaway_cost

CORPUS_V1 = Path(__file__).resolve().parent.parent / "corpus" / "v1"


def write_trace(
    traces_dir: Path,
    snapshots: list[VitalsSnapshot],
    trace_id: str,
) -> str:
    """Write a trace to JSON and return relative path from corpus/v1/."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    data = [s.model_dump(mode="json") for s in snapshots]
    path.write_text(json.dumps(data, indent=2))
    return str(path.relative_to(CORPUS_V1))


def run_confab_collection(
    providers: list[str],
    positives_per_provider: int = 12,
    negatives_per_provider: int = 5,
) -> list[dict]:
    """Collect confabulation traces across multiple providers."""
    entries: list[dict] = []
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    for prov_name in providers:
        provider = get_provider(prov_name)
        print(f"  Confab [{prov_name}]: ", end="", flush=True)

        # Positives (niche topics)
        pos_count = 0
        for i in range(positives_per_provider):
            try:
                run = elicit_confabulation(
                    provider,
                    positive=True,
                    total_steps=6,
                    verify_api_key=api_key,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "confabulation" / "elicited" / "positive",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                is_pos = run.metadata.labels["confabulation"]
                pos_count += 1 if is_pos else 0
                print("+" if is_pos else "-", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        # Negatives (broad topics)
        neg_count = 0
        for i in range(negatives_per_provider):
            try:
                run = elicit_confabulation(
                    provider,
                    positive=False,
                    total_steps=6,
                    verify_api_key=api_key,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "confabulation" / "elicited" / "negative",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                neg_count += 1
                print(".", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        print(f" pos={pos_count} neg={neg_count}")

    return entries


def run_thrash_collection(
    providers: list[str],
    positives_per_provider: int = 12,
    negatives_per_provider: int = 5,
) -> list[dict]:
    """Collect thrash traces across providers."""
    entries: list[dict] = []

    for prov_name in providers:
        provider = get_provider(prov_name)
        print(f"  Thrash [{prov_name}]: ", end="", flush=True)

        pos_count = 0
        for i in range(positives_per_provider):
            try:
                run = elicit_thrash(
                    provider,
                    positive=True,
                    total_steps=8,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "thrash" / "elicited" / "positive",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                is_pos = run.metadata.labels["thrash"]
                pos_count += 1 if is_pos else 0
                print("+" if is_pos else "-", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        neg_count = 0
        for i in range(negatives_per_provider):
            try:
                run = elicit_thrash(
                    provider,
                    positive=False,
                    total_steps=6,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "thrash" / "elicited" / "negative",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                neg_count += 1
                print(".", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        print(f" pos={pos_count} neg={neg_count}")

    return entries


def run_runaway_collection(
    providers: list[str],
    positives_per_provider: int = 12,
    negatives_per_provider: int = 5,
) -> list[dict]:
    """Collect runaway cost traces across providers."""
    entries: list[dict] = []

    for prov_name in providers:
        provider = get_provider(prov_name)
        print(f"  Runaway [{prov_name}]: ", end="", flush=True)

        pos_count = 0
        for i in range(positives_per_provider):
            try:
                run = elicit_runaway_cost(
                    provider,
                    positive=True,
                    total_steps=8,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "runaway_cost" / "elicited" / "positive",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                is_pos = run.metadata.labels["runaway_cost"]
                pos_count += 1 if is_pos else 0
                print("+" if is_pos else "-", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        neg_count = 0
        for i in range(negatives_per_provider):
            try:
                run = elicit_runaway_cost(
                    provider,
                    positive=False,
                    total_steps=6,
                )
                rel_path = write_trace(
                    CORPUS_V1 / "traces" / "runaway_cost" / "elicited" / "negative",
                    run.snapshots,
                    run.trace_id,
                )
                entries.append(run.metadata.as_manifest_entry(rel_path))
                neg_count += 1
                print(".", end="", flush=True)
            except Exception as e:
                print(f"E({e!r})", end="", flush=True)

        print(f" pos={pos_count} neg={neg_count}")

    return entries


def main() -> None:
    # Load .env if available
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    # Use local providers (zero-cost first) + one 9B for cross-provider coverage
    # 27B for primary volume, 9B-a for cross-provider confab requirement
    confab_providers = ["qwen3.5-27b", "qwen3.5-9b-a"]
    thrash_providers = ["qwen3.5-27b", "qwen3.5-9b-b"]
    runaway_providers = ["qwen3.5-27b", "qwen3.5-9b-c"]

    print("=" * 60)
    print("Agent Vitals Bench — Elicitation Collection Runs")
    print("=" * 60)
    print()

    # Load existing manifest
    manifest_path = CORPUS_V1 / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    existing_ids = {e["trace_id"] for e in existing}
    print(f"Existing corpus: {len(existing)} traces")
    print()

    all_new: list[dict] = []

    print("--- Confabulation Collection ---")
    confab_entries = run_confab_collection(
        confab_providers, positives_per_provider=12, negatives_per_provider=5
    )
    all_new.extend(confab_entries)
    print()

    print("--- Thrash Collection ---")
    thrash_entries = run_thrash_collection(
        thrash_providers, positives_per_provider=12, negatives_per_provider=5
    )
    all_new.extend(thrash_entries)
    print()

    print("--- Runaway Cost Collection ---")
    runaway_entries = run_runaway_collection(
        runaway_providers, positives_per_provider=12, negatives_per_provider=5
    )
    all_new.extend(runaway_entries)
    print()

    # Filter duplicates and merge
    new_entries = [e for e in all_new if e["trace_id"] not in existing_ids]
    merged = existing + new_entries

    # Write updated manifest
    manifest_path.write_text(json.dumps(merged, indent=2))

    # Summary
    print("=" * 60)
    print(f"New traces collected: {len(new_entries)}")
    print(f"Total corpus size: {len(merged)}")
    print()

    # Per-detector breakdown of new traces
    for det in ["confabulation", "thrash", "runaway_cost"]:
        pos = sum(1 for e in new_entries if e["labels"].get(det, False))
        neg = sum(1 for e in new_entries if not e["labels"].get(det, False)
                  and det in e.get("metadata", {}).get("generator", "").lower())
        total = sum(1 for e in new_entries
                    if det in e.get("metadata", {}).get("generator", "").lower())
        print(f"  {det:20s}  pos={pos:3d}  total={total:3d}")

    print()
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
