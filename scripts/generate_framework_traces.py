"""Generate per-framework synthetic traces for Sprint 9 cross-framework validation.

Produces traces annotated with framework profiles (langgraph, crewai, dspy).
Each framework's threshold profile changes detection boundaries, so generators
use adjusted parameters to produce traces that reliably trigger (or avoid)
detection under each framework's specific thresholds.

Per-framework threshold overrides (from agent-vitals thresholds.yaml):
  langgraph: loop_consecutive_pct=0.4, burn_rate_multiplier=2.5
  crewai:    burn_rate_multiplier=3.0, token_scale_factor=0.7
  dspy:      loop_consecutive_pct=0.7, stuck_dm_threshold=0.1, stuck=disabled

Target: 40+ traces per framework profile (8+ per detector per framework).

Usage:
    python scripts/generate_framework_traces.py
    python scripts/generate_framework_traces.py --frameworks langgraph crewai
    python scripts/generate_framework_traces.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from agent_vitals.schema import VitalsSnapshot

from generators.base import FRAMEWORK_PROFILES, TraceMetadata
from generators.loop import LoopGenerator
from generators.stuck import StuckGenerator
from generators.confabulation import ConfabGenerator
from generators.thrash import ThrashGenerator
from generators.runaway_cost import RunawayCostGenerator

CORPUS_V1 = Path(__file__).resolve().parent.parent / "corpus" / "v1"


def write_trace(
    traces_dir: Path,
    snapshots: list[VitalsSnapshot],
    trace_id: str,
) -> str:
    """Write a trace and return relative path from corpus/v1/."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    data = [s.model_dump(mode="json") for s in snapshots]
    path.write_text(json.dumps(data, indent=2))
    return str(path.relative_to(CORPUS_V1))


def patch_metadata(meta: TraceMetadata, framework: str) -> TraceMetadata:
    """Return a new TraceMetadata with framework field set."""
    # TraceMetadata is frozen, so reconstruct
    return TraceMetadata(
        trace_id=meta.trace_id,
        generator=meta.generator,
        tier=meta.tier,
        labels=meta.labels,
        params={**meta.params, "framework": framework},
        onset_step=meta.onset_step,
        confidence=meta.confidence,
        notes=f"framework-profile: {framework}",
        framework=framework,
    )


def generate_loop_framework(framework: str) -> list[dict]:
    """Generate loop traces calibrated for a framework profile."""
    gen = LoopGenerator()
    entries: list[dict] = []
    profile = FRAMEWORK_PROFILES.get(framework, {})
    pct = profile.get("loop_consecutive_pct", 0.5)
    seq = 1

    # Positive: adjust loop_length to exceed framework's consecutive_pct
    for total_steps, pattern in product([8, 10, 12, 15], ["exact", "semantic"]):
        # Need loop_length/total_steps > pct for reliable detection
        min_loop = int(total_steps * pct) + 1
        for loop_start in [1, 3]:
            if loop_start + min_loop > total_steps:
                continue
            trace_id = f"loop-{framework[:2]}-{seq:03d}"
            try:
                snapshots, meta = gen.generate(
                    trace_id=trace_id,
                    loop_start=loop_start,
                    loop_length=min_loop,
                    total_steps=total_steps,
                    pattern=pattern,
                    positive=True,
                )
            except ValueError:
                continue
            meta = patch_metadata(meta, framework)
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "loop" / "framework" / "positive",
                snapshots, trace_id,
            )
            entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    # Negative
    for total_steps in [8, 10, 12]:
        trace_id = f"loop-{framework[:2]}-neg-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id, total_steps=total_steps, positive=False,
        )
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "loop" / "framework" / "negative",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    return entries


def generate_stuck_framework(framework: str) -> list[dict]:
    """Generate stuck traces calibrated for a framework profile."""
    gen = StuckGenerator()
    entries: list[dict] = []
    profile = FRAMEWORK_PROFILES.get(framework, {})
    dm_thresh = profile.get("stuck_dm_threshold", 0.15)
    stuck_enabled = profile.get("workflow_stuck_enabled", "research-only")
    seq = 1

    if stuck_enabled == "none":
        # dspy disables stuck — generate traces that would trigger under default
        # but are labeled negative since stuck is disabled for this framework
        for step_count in [6, 8, 10]:
            trace_id = f"stuck-{framework[:2]}-disabled-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id, step_count=step_count,
                stuck_start=3, dm_value=0.05, positive=True,
            )
            # Override label: stuck is disabled for this framework
            labels = {**meta.labels, "stuck": False}
            meta = TraceMetadata(
                trace_id=meta.trace_id, generator=meta.generator,
                labels=labels, params={**meta.params, "framework": framework},
                onset_step=meta.onset_step, confidence=meta.confidence,
                notes=f"framework-profile: {framework} (stuck disabled)",
                framework=framework,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "stuck" / "framework" / "negative",
                snapshots, trace_id,
            )
            entries.append(meta.as_manifest_entry(rel_path))
            seq += 1
        return entries

    # Positive: dm_value well below framework's threshold
    for step_count, dm_value, token_pattern in product(
        [6, 8, 10],
        [0.0, dm_thresh * 0.3, dm_thresh * 0.6],
        ["flat", "slow_rise"],
    ):
        trace_id = f"stuck-{framework[:2]}-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id, step_count=step_count,
                stuck_start=3, dm_value=round(dm_value, 3),
                token_burn_pattern=token_pattern, positive=True,
            )
        except ValueError:
            continue
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "stuck" / "framework" / "positive",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative
    for step_count in [6, 8, 10]:
        trace_id = f"stuck-{framework[:2]}-neg-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id, step_count=step_count, positive=False,
        )
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "stuck" / "framework" / "negative",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    return entries


def generate_confab_framework(framework: str) -> list[dict]:
    """Generate confabulation traces for a framework profile."""
    gen = ConfabGenerator()
    entries: list[dict] = []
    seq = 1

    # Confab thresholds don't vary per framework — same signal calibration
    for total_steps, onset_step, ratio in product(
        [8, 10, 12], [2, 3], [0.1, 0.15, 0.2],
    ):
        if onset_step >= total_steps - 2:
            continue
        trace_id = f"confab-{framework[:2]}-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id, total_steps=total_steps,
                onset_step=onset_step, source_finding_ratio=ratio,
                positive=True,
            )
        except ValueError:
            continue
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "confabulation" / "framework" / "positive",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative
    for total_steps in [8, 10, 12]:
        trace_id = f"confab-{framework[:2]}-neg-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id, total_steps=total_steps, positive=False,
        )
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "confabulation" / "framework" / "negative",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    return entries


def generate_thrash_framework(framework: str) -> list[dict]:
    """Generate thrash traces for a framework profile."""
    gen = ThrashGenerator()
    entries: list[dict] = []
    seq = 1

    # Thrash thresholds don't vary per framework
    for total_steps, onset_step, error_spikes in product(
        [6, 8, 10], [2, 3], [2, 3],
    ):
        if onset_step >= total_steps - 2:
            continue
        trace_id = f"thrash-{framework[:2]}-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id, total_steps=total_steps,
                onset_step=onset_step, error_spikes=error_spikes,
                positive=True,
            )
        except ValueError:
            continue
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "thrash" / "framework" / "positive",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative
    for total_steps in [6, 8, 10]:
        trace_id = f"thrash-{framework[:2]}-neg-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id, total_steps=total_steps, positive=False,
        )
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "thrash" / "framework" / "negative",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    return entries


def generate_runaway_framework(framework: str) -> list[dict]:
    """Generate runaway cost traces calibrated for a framework profile."""
    gen = RunawayCostGenerator()
    entries: list[dict] = []
    profile = FRAMEWORK_PROFILES.get(framework, {})
    burn_mult = profile.get("burn_rate_multiplier", 2.5)
    seq = 1

    # Positive: burn_rate must exceed framework's burn_rate_multiplier
    for total_steps, onset_step, cost_growth in product(
        [6, 8, 10], [2, 3], ["linear", "quadratic"],
    ):
        if onset_step >= total_steps - 2:
            continue
        # Set burn_rate well above the framework's multiplier threshold
        burn_rate = burn_mult * 1.5
        trace_id = f"runaway-{framework[:2]}-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id, total_steps=total_steps,
                onset_step=onset_step, cost_growth=cost_growth,
                burn_rate=burn_rate, positive=True,
            )
        except ValueError:
            continue
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "runaway_cost" / "framework" / "positive",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative
    for total_steps in [6, 8, 10]:
        trace_id = f"runaway-{framework[:2]}-neg-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id, total_steps=total_steps, positive=False,
        )
        meta = patch_metadata(meta, framework)
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "runaway_cost" / "framework" / "negative",
            snapshots, trace_id,
        )
        entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    return entries


GENERATORS = {
    "loop": generate_loop_framework,
    "stuck": generate_stuck_framework,
    "confabulation": generate_confab_framework,
    "thrash": generate_thrash_framework,
    "runaway_cost": generate_runaway_framework,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-framework synthetic traces"
    )
    parser.add_argument(
        "--frameworks", nargs="+",
        default=list(FRAMEWORK_PROFILES),
        choices=list(FRAMEWORK_PROFILES),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Per-framework synthetic trace generation plan:")
        print(f"  Frameworks: {args.frameworks}")
        print(f"  Detectors: {list(GENERATORS)}")
        print("  Target: 40+ per framework (8+ per detector per framework)")
        for fw in args.frameworks:
            profile = FRAMEWORK_PROFILES.get(fw, {})
            print(f"\n  {fw}:")
            for k, v in profile.items():
                print(f"    {k}: {v}")
        return

    # Load existing manifest and preserve non-framework entries
    manifest_path = CORPUS_V1 / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []

    # Remove old framework traces to avoid duplicates on re-run
    old_fw_ids = {
        e["trace_id"] for e in manifest
        if e.get("metadata", {}).get("framework") in args.frameworks
    }
    manifest = [e for e in manifest if e["trace_id"] not in old_fw_ids]
    if old_fw_ids:
        print(f"Removed {len(old_fw_ids)} old framework traces from manifest")

    new_entries: list[dict] = []

    for framework in args.frameworks:
        print(f"\n{'='*60}")
        print(f"Framework: {framework}")
        print(f"{'='*60}")
        fw_total = 0

        for detector, gen_func in GENERATORS.items():
            entries = gen_func(framework)
            pos = sum(1 for e in entries if e["labels"].get(detector, False))
            neg = len(entries) - pos
            print(f"  {detector:20s}  pos={pos:3d}  neg={neg:3d}  total={len(entries)}")
            new_entries.extend(entries)
            fw_total += len(entries)

        print(f"  {'TOTAL':20s}  {fw_total}")

    manifest.extend(new_entries)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nNew framework traces: {len(new_entries)}")
    print(f"Total manifest entries: {len(manifest)}")

    # Per-framework summary
    print("\nPer-framework counts:")
    for fw in args.frameworks:
        count = sum(
            1 for e in new_entries
            if e.get("metadata", {}).get("framework") == fw
        )
        print(f"  {fw}: {count} traces")


if __name__ == "__main__":
    main()
