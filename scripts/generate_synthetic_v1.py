"""Generate synthetic corpus v1 via parameter sweep across all 5 generators.

Produces 40+ positives and 40+ negatives per detector, written to
corpus/v1/traces/{detector}/{positive,negative}/ with manifest entries.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from agent_vitals.schema import VitalsSnapshot

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
    """Write a trace to JSONL and return relative path from corpus/v1/."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    data = [s.model_dump(mode="json") for s in snapshots]
    path.write_text(json.dumps(data, indent=2))
    return str(path.relative_to(CORPUS_V1))


def generate_loop_sweep() -> list[dict]:
    """Generate loop traces across parameter sweep."""
    gen = LoopGenerator()
    manifest_entries: list[dict] = []
    seq = 1

    # Positive sweep
    for loop_start, loop_length, total_steps, pattern in product(
        [1, 3, 5, 8],        # early, mid, late, near-end
        [2, 3, 4, 6],        # minimal to long
        [8, 12, 15, 20],     # various lengths
        ["exact", "semantic", "partial"],
    ):
        if loop_start + loop_length > total_steps:
            continue
        if total_steps < 3:
            continue

        trace_id = f"loop-syn-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                loop_start=loop_start,
                loop_length=loop_length,
                total_steps=total_steps,
                pattern=pattern,
                positive=True,
            )
        except ValueError:
            continue

        rel_path = write_trace(
            CORPUS_V1 / "traces" / "loop" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative traces (healthy, no loop)
    for total_steps in [6, 8, 10, 12, 15, 20]:
        for neg_seq in range(4):
            trace_id = f"loop-syn-neg-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                positive=False,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "loop" / "negative",
                snapshots, trace_id,
            )
            manifest_entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    return manifest_entries


def generate_stuck_sweep() -> list[dict]:
    """Generate stuck traces across parameter sweep."""
    gen = StuckGenerator()
    manifest_entries: list[dict] = []
    seq = 1

    # Positive sweep
    for step_count, stuck_start, dm_value, token_pattern in product(
        [5, 8, 10, 12],
        [2, 3, 5],
        [0.0, 0.05, 0.10],   # All below threshold (0.15)
        ["flat", "slow_rise", "fast_rise"],
    ):
        if stuck_start >= step_count - 1:
            continue

        trace_id = f"stuck-syn-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                step_count=step_count,
                stuck_start=stuck_start,
                dm_value=dm_value,
                token_burn_pattern=token_pattern,
                positive=True,
            )
        except ValueError:
            continue

        rel_path = write_trace(
            CORPUS_V1 / "traces" / "stuck" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Boundary cases: dm_value at/above threshold (should NOT trigger)
    for dm_val in [0.15, 0.20, 0.30]:
        trace_id = f"stuck-syn-boundary-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id,
            step_count=8,
            stuck_start=3,
            dm_value=dm_val,
            positive=True,  # Labeled positive but may not fire
        )
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "stuck" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative traces
    for step_count in [5, 8, 10, 12, 15]:
        for neg_seq in range(5):
            trace_id = f"stuck-syn-neg-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                step_count=step_count,
                positive=False,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "stuck" / "negative",
                snapshots, trace_id,
            )
            manifest_entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    return manifest_entries


def generate_confab_sweep() -> list[dict]:
    """Generate confabulation traces across parameter sweep."""
    gen = ConfabGenerator()
    manifest_entries: list[dict] = []
    seq = 1

    # Positive sweep
    for total_steps, onset_step, ratio, confidence_infl in product(
        [8, 10, 12, 15],
        [2, 3, 5],
        [0.1, 0.15, 0.2, 0.25],    # All below floor (0.3)
        [0.0, 0.2, 0.4],
    ):
        if onset_step >= total_steps - 2:
            continue

        trace_id = f"confab-syn-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                onset_step=onset_step,
                source_finding_ratio=ratio,
                confidence_inflation=confidence_infl,
                positive=True,
            )
        except ValueError:
            continue

        rel_path = write_trace(
            CORPUS_V1 / "traces" / "confabulation" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Boundary: ratio at/above floor (should NOT trigger)
    for ratio in [0.3, 0.35, 0.5]:
        trace_id = f"confab-syn-boundary-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id,
            total_steps=10,
            onset_step=3,
            source_finding_ratio=ratio,
            positive=True,
        )
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "confabulation" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative traces
    for total_steps in [6, 8, 10, 12, 15]:
        for neg_seq in range(5):
            trace_id = f"confab-syn-neg-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                positive=False,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "confabulation" / "negative",
                snapshots, trace_id,
            )
            manifest_entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    return manifest_entries


def generate_thrash_sweep() -> list[dict]:
    """Generate thrash traces across parameter sweep."""
    gen = ThrashGenerator()
    manifest_entries: list[dict] = []
    seq = 1

    # Positive sweep
    for total_steps, onset_step, error_spikes, refinement_growth in product(
        [6, 8, 10, 12],
        [2, 3],
        [2, 3, 5],
        [1, 2, 3],
    ):
        if onset_step >= total_steps - 2:
            continue

        trace_id = f"thrash-syn-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                onset_step=onset_step,
                error_spikes=error_spikes,
                refinement_growth=refinement_growth,
                positive=True,
            )
        except ValueError:
            continue

        rel_path = write_trace(
            CORPUS_V1 / "traces" / "thrash" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative traces
    for total_steps in [6, 8, 10, 12]:
        for neg_seq in range(6):
            trace_id = f"thrash-syn-neg-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                positive=False,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "thrash" / "negative",
                snapshots, trace_id,
            )
            manifest_entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    return manifest_entries


def generate_runaway_sweep() -> list[dict]:
    """Generate runaway cost traces across parameter sweep."""
    gen = RunawayCostGenerator()
    manifest_entries: list[dict] = []
    seq = 1

    # Positive sweep
    for total_steps, onset_step, cost_growth, burn_rate in product(
        [6, 8, 10, 12],
        [2, 3],
        ["linear", "quadratic", "step"],
        [4.0, 6.0, 8.0],     # Well above threshold (3.0) for reliable detection
    ):
        if onset_step >= total_steps - 2:
            continue

        trace_id = f"runaway-syn-{seq:03d}"
        try:
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                onset_step=onset_step,
                cost_growth=cost_growth,
                burn_rate=burn_rate,
                positive=True,
            )
        except ValueError:
            continue

        rel_path = write_trace(
            CORPUS_V1 / "traces" / "runaway_cost" / "positive",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Boundary: burn_rate at or below threshold (should NOT trigger)
    for br in [2.0, 2.5, 3.0]:
        trace_id = f"runaway-syn-boundary-{seq:03d}"
        snapshots, meta = gen.generate(
            trace_id=trace_id,
            total_steps=8,
            onset_step=3,
            cost_growth="linear",
            burn_rate=br,
            positive=False,  # Below/at threshold — negative examples
        )
        rel_path = write_trace(
            CORPUS_V1 / "traces" / "runaway_cost" / "boundary",
            snapshots, trace_id,
        )
        manifest_entries.append(meta.as_manifest_entry(rel_path))
        seq += 1

    # Negative traces
    for total_steps in [6, 8, 10, 12]:
        for neg_seq in range(6):
            trace_id = f"runaway-syn-neg-{seq:03d}"
            snapshots, meta = gen.generate(
                trace_id=trace_id,
                total_steps=total_steps,
                positive=False,
            )
            rel_path = write_trace(
                CORPUS_V1 / "traces" / "runaway_cost" / "negative",
                snapshots, trace_id,
            )
            manifest_entries.append(meta.as_manifest_entry(rel_path))
            seq += 1

    return manifest_entries


def main() -> None:
    print("Generating synthetic corpus v1...")

    all_entries: list[dict] = []

    generators = [
        ("loop", generate_loop_sweep),
        ("stuck", generate_stuck_sweep),
        ("confabulation", generate_confab_sweep),
        ("thrash", generate_thrash_sweep),
        ("runaway_cost", generate_runaway_sweep),
    ]

    for name, gen_func in generators:
        entries = gen_func()
        pos = sum(1 for e in entries if e["labels"].get(name, False))
        neg = sum(1 for e in entries if not e["labels"].get(name, False))
        print(f"  {name:20s}  pos={pos:3d}  neg={neg:3d}  total={len(entries)}")
        all_entries.extend(entries)

    # Write manifest
    manifest_path = CORPUS_V1 / "manifest.json"
    manifest_path.write_text(json.dumps(all_entries, indent=2))

    print(f"\nTotal traces: {len(all_entries)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
