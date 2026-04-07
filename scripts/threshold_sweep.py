#!/usr/bin/env python3
"""Threshold sensitivity sweep for Paper C.

Runs the evaluator across a range of threshold values (±20% around defaults)
for each detector's key thresholds. Produces per-detector sensitivity data
showing how P_lb and R_lb change as thresholds shift.

The sweep answers: "How fragile are our gate passes?" If a small threshold
change flips a detector from HARD GATE to NO-GO, that detector is brittle
and needs a wider margin or more traces.

Output:
  - reports/threshold-sensitivity.md  (human-readable summary)
  - reports/threshold-sensitivity.json (machine-readable sweep data)

Usage:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --corpus v1 --steps 11
    python scripts/threshold_sweep.py --detectors confabulation runaway_cost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_vitals import VitalsConfig

from evaluator.runner import run_evaluation, DETECTORS
from evaluator.gate import MIN_PRECISION_LB, MIN_RECALL_LB, MIN_POSITIVES

REPORTS_DIR = PROJECT_ROOT / "reports"

# Key thresholds per detector and their default values + sweep ranges.
# Each entry: (config_field, default_value, low_pct, high_pct)
SWEEP_CONFIG: dict[str, list[dict]] = {
    "loop": [
        {
            "field": "loop_consecutive_pct",
            "default": 0.5,
            "low": 0.3,
            "high": 0.7,
            "description": "Fraction of trace that must plateau for loop detection",
        },
        {
            "field": "loop_similarity_threshold",
            "default": 0.8,
            "low": 0.6,
            "high": 0.95,
            "description": "Output similarity threshold for loop detection",
        },
    ],
    "stuck": [
        {
            "field": "stuck_dm_threshold",
            "default": 0.15,
            "low": 0.05,
            "high": 0.25,
            "description": "Directional momentum threshold for stuck detection",
        },
        {
            "field": "stuck_cv_threshold",
            "default": 0.3,
            "low": 0.15,
            "high": 0.45,
            "description": "Coefficient of variation threshold for stuck detection",
        },
    ],
    "confabulation": [
        {
            "field": "source_finding_ratio_floor",
            "default": 0.3,
            "low": 0.15,
            "high": 0.45,
            "description": "Source-to-finding ratio floor for confab detection",
        },
    ],
    "thrash": [
        {
            "field": "thrash_error_threshold",
            "default": 1,
            "low": 1,
            "high": 5,
            "description": "Error count threshold for thrash detection",
        },
    ],
    "runaway_cost": [
        {
            "field": "burn_rate_multiplier",
            "default": 2.5,
            "low": 1.5,
            "high": 5.0,
            "description": "Token burn rate multiplier threshold for runaway detection",
        },
    ],
}


def sweep_threshold(
    field: str,
    values: list[float],
    corpus_version: str,
    detector: str,
) -> list[dict]:
    """Run evaluator for each threshold value and return results."""
    results = []
    base_config = VitalsConfig.from_yaml()

    for val in values:
        # Create config with this threshold override
        overrides = {field: val}
        config = VitalsConfig.from_dict({
            **{k: getattr(base_config, k)
               for k in base_config.__dataclass_fields__
               if k != "framework_profiles"},
            **overrides,
        })

        eval_result = run_evaluation(
            corpus_version=corpus_version,
            detectors=(detector,),
            config=config,
        )

        metrics = eval_result.detector_metrics.get(detector)
        gate = eval_result.gate_results.get(detector, {})

        if metrics:
            p_lb, p_ub = metrics.precision_ci
            r_lb, r_ub = metrics.recall_ci
            results.append({
                "threshold_value": round(val, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "precision_lb": round(p_lb, 4),
                "recall_lb": round(r_lb, 4),
                "total_positives": metrics.total_positives,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "gate_status": gate.get("status", "UNKNOWN"),
                "gate_passed": gate.get("passed", False),
            })

    return results


def generate_report(
    all_results: dict[str, dict[str, list[dict]]],
    corpus_version: str,
) -> str:
    """Generate markdown sensitivity report."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Threshold Sensitivity Sweep")
    lines.append("")
    lines.append(f"**Corpus:** {corpus_version}")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Gate thresholds:** P_lb >= {MIN_PRECISION_LB}, R_lb >= {MIN_RECALL_LB}, "
                 f"min {MIN_POSITIVES} positives")
    lines.append("")

    # Executive summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Detector | Threshold | Default | Safe Range | Brittle? |")
    lines.append("|----------|-----------|---------|------------|----------|")

    for detector, thresholds in all_results.items():
        for field, results in thresholds.items():
            if not results:
                continue
            # Find default value
            default_val = next(
                (r["threshold_value"] for r in results
                 if any(s["field"] == field and abs(s["default"] - r["threshold_value"]) < 0.001
                        for s in SWEEP_CONFIG.get(detector, []))),
                results[len(results) // 2]["threshold_value"],
            )
            # Find safe range (where gate passes)
            passing = [r for r in results if r["gate_passed"]]
            if passing:
                safe_low = min(r["threshold_value"] for r in passing)
                safe_high = max(r["threshold_value"] for r in passing)
                safe_range = f"[{safe_low:.3f}, {safe_high:.3f}]"
                # Brittle if safe range < 20% of sweep range
                sweep_range = max(r["threshold_value"] for r in results) - min(
                    r["threshold_value"] for r in results
                )
                safe_width = safe_high - safe_low
                brittle = "YES" if (safe_width / sweep_range < 0.3) else "no"
            else:
                safe_range = "NONE"
                brittle = "YES"

            lines.append(
                f"| {detector} | {field} | {default_val:.3f} "
                f"| {safe_range} | {brittle} |"
            )

    lines.append("")

    # Per-detector detail
    for detector, thresholds in all_results.items():
        for field, results in thresholds.items():
            if not results:
                continue
            lines.append(f"## {detector}: {field}")
            lines.append("")
            lines.append("| Value | P | R | F1 | P_lb | R_lb | TP | FP | FN | Status |")
            lines.append("|-------|---|---|----|----- |------|----|----|----| ------|")

            for r in results:
                status = "PASS" if r["gate_passed"] else "FAIL"
                lines.append(
                    f"| {r['threshold_value']:.3f} "
                    f"| {r['precision']:.3f} "
                    f"| {r['recall']:.3f} "
                    f"| {r['f1']:.3f} "
                    f"| {r['precision_lb']:.3f} "
                    f"| {r['recall_lb']:.3f} "
                    f"| {r['tp']} "
                    f"| {r['fp']} "
                    f"| {r['fn']} "
                    f"| {status} |"
                )
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Threshold sensitivity sweep")
    parser.add_argument("--corpus", default="v1", help="Corpus version")
    parser.add_argument(
        "--detectors", nargs="+", default=None,
        help="Detectors to sweep (default: all with sweep config)",
    )
    parser.add_argument(
        "--steps", type=int, default=11,
        help="Number of threshold values to test per sweep (default: 11)",
    )
    args = parser.parse_args()

    target_detectors = args.detectors or [
        d for d in DETECTORS if SWEEP_CONFIG.get(d)
    ]

    all_results: dict[str, dict[str, list[dict]]] = {}

    for detector in target_detectors:
        configs = SWEEP_CONFIG.get(detector, [])
        if not configs:
            print(f"  {detector}: no sweep thresholds configured, skipping")
            continue

        all_results[detector] = {}

        for cfg in configs:
            field = cfg["field"]
            low = cfg["low"]
            high = cfg["high"]

            # Generate evenly-spaced values.
            # Integer fields (e.g. thrash_error_threshold) use integer steps.
            is_int = isinstance(cfg["default"], int)
            if is_int:
                values = list(range(int(low), int(high) + 1))
            else:
                step_size = (high - low) / (args.steps - 1) if args.steps > 1 else 0
                values = [low + i * step_size for i in range(args.steps)]

            print(f"  {detector}/{field}: sweeping {len(values)} values [{low:.3f}, {high:.3f}]")

            results = sweep_threshold(field, values, args.corpus, detector)
            all_results[detector][field] = results

            # Quick summary
            passing = sum(1 for r in results if r["gate_passed"])
            print(f"    → {passing}/{len(results)} pass HARD GATE")

    # Generate reports
    REPORTS_DIR.mkdir(exist_ok=True)

    report_md = generate_report(all_results, args.corpus)
    md_path = REPORTS_DIR / "threshold-sensitivity.md"
    md_path.write_text(report_md)
    print(f"\nMarkdown report: {md_path}")

    json_data = {
        "corpus_version": args.corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_thresholds": {
            "min_precision_lb": MIN_PRECISION_LB,
            "min_recall_lb": MIN_RECALL_LB,
            "min_positives": MIN_POSITIVES,
        },
        "sweeps": all_results,
    }
    json_path = REPORTS_DIR / "threshold-sensitivity.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"JSON report: {json_path}")

    # Print summary
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
