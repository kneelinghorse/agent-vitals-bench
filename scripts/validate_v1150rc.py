"""AV-S09-M03 validation driver for agent-vitals v1.15.0rc.

Runs the bench validation cycle for the v1.15.0 Hopfield early-detection
informational marker per the gating asks in inbound message 6eb0ee48:

  1. Replay-stability: confirm v1.15.0rc with hopfield_enabled=False is
     bit-identical to the v1.14.2 baseline cross-framework artifacts.
  2. Bit-identical contract: same check with hopfield_enabled=True. The
     marker is additive by design (does not flip detector flags), so cells
     must STILL be bit-identical even when the marker is firing.
  3. Marker firing analysis: at cutoffs 3 and 5, what fraction of traces
     with each per-detector positive label have hopfield_override_active=
     True under the v1.15.0rc public API? Compare against the bench p3/p5
     grid from the comparative report and flag >5% relative divergence.

Run this script TWICE:
  - From the tda-experiment Python 3.12 venv (has TDA backend, no
    onnxruntime): pass --hopfield-enabled false --runtime-modes default,tda
    to cover all 8 cells under the no-marker baseline.
  - From the bench main venv (has onnxruntime via [hopfield] extra, no TDA):
    pass --hopfield-enabled true --runtime-modes default to cover the
    4 default-mode cells with the marker firing.

Marker firing analysis (Q3) only runs when --hopfield-enabled true is set
(it requires onnxruntime).

Outputs are written to /tmp/v1150rc-* paths (not committed to reports/) so
the canonical eval-cross-framework-v1.{md,json} artifacts stay clean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_vitals.detection.tda import is_tda_available  # noqa: E402

from evaluator.runner import DETECTORS, load_manifest, load_trace  # noqa: E402

from scripts.regen_cross_framework import (  # noqa: E402
    RUNTIME_MODES,
    CellSummary,
    run_grid,
)

BASELINE_PATH = REPO_ROOT / "reports" / "eval-cross-framework-v1.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_baseline() -> dict[tuple[str, str], dict[str, Any]]:
    """Load the v1.14.2 cross-framework baseline keyed by (profile, mode)."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline {BASELINE_PATH} missing — regen against v1.14.2 first.")
    payload = json.loads(BASELINE_PATH.read_text())
    return {(c["profile"], c["runtime_mode"]): c for c in payload["cells"]}


def _diff_cells_against_baseline(
    cells: list[CellSummary],
    baseline: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a list of (cell, detector) pairs that differ in TP/FP/FN/TN."""
    deltas: list[dict[str, Any]] = []
    for cell in cells:
        key = (cell.profile, cell.runtime_mode)
        if key not in baseline:
            deltas.append(
                {"reason": "baseline_missing", "cell": cell.profile, "mode": cell.runtime_mode}
            )
            continue
        bcell = baseline[key]
        for detector in DETECTORS:
            new = cell.detector_metrics[detector]
            old = bcell["detector_metrics"][detector]
            if (
                new["tp"] != old["tp"]
                or new["fp"] != old["fp"]
                or new["fn"] != old["fn"]
                or new["tn"] != old["tn"]
            ):
                deltas.append(
                    {
                        "profile": cell.profile,
                        "runtime_mode": cell.runtime_mode,
                        "detector": detector,
                        "old": {k: old[k] for k in ("tp", "fp", "fn", "tn")},
                        "new": {k: new[k] for k in ("tp", "fp", "fn", "tn")},
                    }
                )
    return deltas


# ---------------------------------------------------------------------------
# Marker firing analysis (Q3)
# ---------------------------------------------------------------------------


def _run_marker_analysis(
    *,
    cutoffs: tuple[int, ...] = (3, 5),
    min_confidence: float = 0.8,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> dict[str, Any]:
    """Score every v1 trace through the v1.15.0 Hopfield public API at each
    cutoff, count marker firings per detector at each candidate threshold,
    and report TP/FP cell metrics broken down by label.

    Uses the public ``agent_vitals.detection.hopfield.predict`` (or
    ``hopfield_override_fires``) function so this validation runs against the
    same code path the v1.15.0 production wiring goes through, not against
    the bench prototype directly.
    """
    from agent_vitals.detection import hopfield as av_hopfield

    if not av_hopfield.is_hopfield_available():
        raise RuntimeError(
            "agent_vitals.detection.hopfield reports backend unavailable — "
            "marker analysis needs onnxruntime importable. Install via "
            "pip install '/tmp/av_wheel_v1150/agent_vitals-1.15.0-py3-none-any.whl[hopfield]'."
        )

    manifest = load_manifest("v1")
    eligible = [
        e
        for e in manifest
        if float(e.get("metadata", {}).get("confidence", 0.0) or 0.0) >= min_confidence
    ]
    print(
        f"[marker] eligible traces (min_confidence>={min_confidence}): {len(eligible)}",
        flush=True,
    )

    snapshots_by_path: dict[str, list[Any]] = {}
    for entry in eligible:
        path = str(entry["path"])
        if path not in snapshots_by_path:
            snapshots_by_path[path] = load_trace("v1", path)

    # marker[cutoff][detector][threshold] -> {tp, fp, fn, tn, label_pos, label_neg}
    marker: dict[int, dict[str, dict[float, dict[str, int]]]] = {
        cutoff: {
            d: {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds} for d in DETECTORS
        }
        for cutoff in cutoffs
    }
    label_counts = {d: {"pos": 0, "neg": 0} for d in DETECTORS}
    detector_seen = {d: 0 for d in DETECTORS}

    for entry in eligible:
        snapshots = snapshots_by_path[str(entry["path"])]
        labels = {d: bool(entry["labels"].get(d, False)) for d in DETECTORS}
        for d, lbl in labels.items():
            if lbl:
                label_counts[d]["pos"] += 1
            else:
                label_counts[d]["neg"] += 1

        for cutoff in cutoffs:
            truncated = snapshots[:cutoff]
            if not truncated:
                continue
            for d in DETECTORS:
                try:
                    # public API: predict(snapshots, detector) -> HopfieldPrediction | None
                    pred = av_hopfield.predict(truncated, d)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[marker] WARN predict({entry.get('trace_id')}, {d}) failed: {exc}",
                        flush=True,
                    )
                    continue
                if pred is None:
                    continue
                detector_seen[d] += 1
                proba = float(pred.probability)
                lbl = labels[d]
                for t in thresholds:
                    fired = proba >= t
                    cell = marker[cutoff][d][t]
                    if fired and lbl:
                        cell["tp"] += 1
                    elif fired and not lbl:
                        cell["fp"] += 1
                    elif not fired and lbl:
                        cell["fn"] += 1
                    else:
                        cell["tn"] += 1

    # Compute precision/recall/F1 per cell
    summary: dict[str, Any] = {
        "label_counts": label_counts,
        "thresholds_evaluated": list(thresholds),
        "cutoffs_evaluated": list(cutoffs),
        "by_cutoff": {},
    }
    for cutoff in cutoffs:
        per_detector: dict[str, dict[str, Any]] = {}
        for d in DETECTORS:
            curve: dict[str, dict[str, float]] = {}
            for t in thresholds:
                c = marker[cutoff][d][t]
                tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                curve[f"{t:.2f}"] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f1, 4),
                }
            per_detector[d] = curve
        summary["by_cutoff"][cutoff] = per_detector
    return summary


def _compare_marker_against_p_grid(
    marker_summary: dict[str, Any],
    *,
    bench_grid: dict[int, dict[str, float]],
    threshold_for_compare: float = 0.5,
    relative_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare the v1.15.0 marker firing F1 at sigmoid 0.5 against the bench
    five-paradigm comparative report's per-prefix F1 grid. Flag any
    detector × cutoff cell that diverges by >tolerance relative.
    """
    flags: list[dict[str, Any]] = []
    key = f"{threshold_for_compare:.2f}"
    for cutoff, expected_per_detector in bench_grid.items():
        new_per_detector = marker_summary["by_cutoff"].get(cutoff, {})
        for d, expected_f1 in expected_per_detector.items():
            curve = new_per_detector.get(d, {})
            new_cell = curve.get(key)
            if not new_cell:
                continue
            new_f1 = float(new_cell["f1"])
            if expected_f1 == 0:
                continue
            relative = abs(new_f1 - expected_f1) / expected_f1
            flags.append(
                {
                    "cutoff": cutoff,
                    "detector": d,
                    "bench_grid_f1": round(expected_f1, 4),
                    "v1150_marker_f1": round(new_f1, 4),
                    "delta": round(new_f1 - expected_f1, 4),
                    "relative_pct": round(100 * relative, 2),
                    "exceeds_tolerance": relative > relative_tolerance,
                }
            )
    return flags


# Bench comparative report p3/p5 numbers (sigmoid 0.5, from
# reports/eval-five-paradigm-comparative.json hopfield rows). The agent-vitals
# v1.15.0 marker should reproduce these within rounding because it uses the
# same artifacts via ONNX export with verified parity at 6.2e-6.
BENCH_P_GRID: dict[int, dict[str, float]] = {
    3: {
        "loop": 0.936,
        "stuck": 0.890,
        "confabulation": 0.901,
        "thrash": 0.941,
        "runaway_cost": 0.836,
    },
    5: {
        "loop": 0.986,
        "stuck": 0.899,
        "confabulation": 0.920,
        "thrash": 1.000,
        "runaway_cost": 1.000,
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--hopfield-enabled",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=False,
        help="Set VitalsConfig.hopfield_enabled (true/false; default false)",
    )
    parser.add_argument(
        "--runtime-modes",
        default="default,tda",
        help="Comma-separated runtime modes to run (default,tda or default)",
    )
    parser.add_argument(
        "--marker-analysis",
        action="store_true",
        help="Also run the cell-level marker firing analysis (Q3). Requires "
        "onnxruntime importable.",
    )
    parser.add_argument(
        "--out-prefix",
        default="/tmp/v1150rc-validation",
        help="Output path prefix (default /tmp/v1150rc-validation)",
    )
    args = parser.parse_args()

    runtime_modes_tuple = tuple(m.strip() for m in args.runtime_modes.split(","))
    for m in runtime_modes_tuple:
        if m not in RUNTIME_MODES:
            parser.error(f"unknown runtime mode {m!r}; allowed: {RUNTIME_MODES}")

    print(
        f"[validate_v1150rc] hopfield_enabled={args.hopfield_enabled} "
        f"runtime_modes={runtime_modes_tuple} marker_analysis={args.marker_analysis}",
        flush=True,
    )
    print(f"[validate_v1150rc] tda_available={is_tda_available()}", flush=True)

    cells = run_grid(
        runtime_modes=runtime_modes_tuple,
        hopfield_enabled=args.hopfield_enabled,
    )

    baseline = _load_baseline()
    deltas = _diff_cells_against_baseline(cells, baseline)

    print(f"\n[validate_v1150rc] cells run: {len(cells)}")
    print(f"[validate_v1150rc] (cell × detector) deltas vs baseline: {len(deltas)}")
    if deltas:
        for d in deltas[:20]:
            print(f"  - {d}")
    else:
        print("  (zero deltas — bit-identical to v1.14.2 baseline)")

    output: dict[str, Any] = {
        "hopfield_enabled": args.hopfield_enabled,
        "runtime_modes": list(runtime_modes_tuple),
        "cells_run": len(cells),
        "deltas_vs_v14x_baseline": deltas,
        "bit_identical": len(deltas) == 0,
        "cells": [c.as_dict() for c in cells],
    }

    if args.marker_analysis:
        print("\n[validate_v1150rc] running marker firing analysis (Q3)...", flush=True)
        marker_summary = _run_marker_analysis()
        flags = _compare_marker_against_p_grid(marker_summary, bench_grid=BENCH_P_GRID)
        output["marker_analysis"] = marker_summary
        output["bench_p_grid_comparison"] = flags
        violations = [f for f in flags if f["exceeds_tolerance"]]
        print(
            f"[validate_v1150rc] marker analysis: {len(flags)} cells compared, "
            f"{len(violations)} exceed 5% relative tolerance vs bench grid"
        )
        if violations:
            for v in violations:
                print(f"  ! {v}")

    suffix = "hopfield_on" if args.hopfield_enabled else "hopfield_off"
    out_path = Path(f"{args.out_prefix}-{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n[validate_v1150rc] wrote {out_path}")

    return 0 if len(deltas) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
