#!/usr/bin/env python3
"""Runaway_cost FP sensitivity analysis for agent-vitals upstream coordination.

Answers four questions from agent-vitals (AV-S10-M01 intel request):

1. Coverage at firing step — what fraction of the 52 full-corpus FPs have
   coverage >= 0.50 at the burn_rate_anomaly firing step?
2. Findings_delta at firing step — what fraction have delta >= 2?
3. Burn_rate ratio distribution — histogram at {2.5, 3.0, 3.5, 4.0}
4. Safety check — does any TP have coverage >= 0.50 at the firing step?

Also evaluates the three proposed rule changes:
  Option A: Coverage suppression at >= 0.50 (currently >= 0.95)
  Option B: Raise multiplier from 2.5 to 3.0
  Option C: Minimum baseline depth >= 3

Output:
  reports/runaway-fp-sensitivity.md
  reports/runaway-fp-sensitivity.json

Usage:
    python scripts/runaway_fp_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_vitals import VitalsConfig
from agent_vitals.schema import VitalsSnapshot

from evaluator.runner import load_manifest, load_trace, replay_trace, DETECTORS


@dataclass
class FiringAnalysis:
    """Analysis of a single trace at the burn_rate_anomaly firing step."""

    trace_id: str
    label: bool  # True = positive, False = negative
    classification: str  # TP or FP
    firing_step: int  # snapshot index where burn_rate_anomaly fires
    total_steps: int
    coverage_at_fire: float
    findings_delta_at_fire: float
    token_delta_at_fire: float
    burn_rate_ratio: float  # actual ratio / baseline
    baseline_value: float
    baseline_depth: int  # number of baseline samples


def _deltas(values: list[float]) -> list[float]:
    """Compute step-over-step deltas (mirrors agent-vitals _deltas)."""
    if len(values) < 2:
        return []
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def analyze_firing_step(
    snapshots: list[VitalsSnapshot],
    cfg: VitalsConfig,
) -> FiringAnalysis | None:
    """Replay burn_rate_anomaly logic step-by-step to find the firing point.

    Returns analysis of the first step where the detector fires, or None.
    Mirrors _burn_rate_anomaly_confidence from agent-vitals loop.py.
    """
    scale = max(0.01, float(cfg.token_scale_factor))
    multiplier = float(cfg.burn_rate_multiplier)

    # Build cumulative arrays as the detector sees them
    token_totals = []
    findings_counts = []
    coverage_scores_list = []

    for snap in snapshots:
        token_totals.append(float(snap.signals.total_tokens))
        findings_counts.append(float(snap.signals.findings_count))
        coverage_scores_list.append(float(snap.signals.coverage_score))

    # Check each step (detector sees history up to and including current step)
    for step_idx in range(1, len(snapshots)):
        # The detector operates on deltas of the series seen so far
        series_tokens = token_totals[: step_idx + 1]
        series_findings = findings_counts[: step_idx + 1]
        series_coverage = coverage_scores_list[: step_idx + 1]

        token_deltas = _deltas(series_tokens)
        findings_deltas = _deltas(series_findings)

        if not token_deltas or not findings_deltas:
            continue

        # Coverage check (>= 0.95 suppresses)
        if series_coverage[-1] >= 0.95:
            continue

        current_tokens = token_deltas[-1] * scale
        current_findings = findings_deltas[-1]
        if current_tokens <= 0.0:
            continue

        # Build baseline ratios from prior steps
        baseline_ratios: list[float] = []
        baseline_token_deltas_list: list[float] = []
        for idx in range(len(findings_deltas) - 1):
            bt = token_deltas[idx] * scale
            bf = findings_deltas[idx]
            if bt <= 0.0:
                continue
            baseline_token_deltas_list.append(bt)
            if bf > 0.0:
                baseline_ratios.append(bt / bf)

        if not baseline_ratios:
            continue

        baseline = float(mean(baseline_ratios))
        if baseline <= 0.0:
            continue

        ratio = math.inf if current_findings <= 0.0 else float(
            current_tokens / current_findings
        )
        if ratio <= multiplier * baseline:
            continue

        # Zero-findings token check
        if current_findings <= 0.0 and baseline_token_deltas_list:
            token_baseline = float(mean(baseline_token_deltas_list))
            if current_tokens <= multiplier * token_baseline:
                continue

        # Detector fires at this step
        return FiringAnalysis(
            trace_id="",  # filled by caller
            label=False,  # filled by caller
            classification="",  # filled by caller
            firing_step=step_idx,
            total_steps=len(snapshots),
            coverage_at_fire=series_coverage[-1],
            findings_delta_at_fire=current_findings / scale,  # undo scale
            token_delta_at_fire=current_tokens / scale,
            burn_rate_ratio=ratio / baseline if baseline > 0 else math.inf,
            baseline_value=baseline,
            baseline_depth=len(baseline_ratios),
        )

    return None


def simulate_option_a(analyses: list[FiringAnalysis], cov_threshold: float = 0.50) -> dict:
    """Option A: suppress when coverage >= threshold at firing step."""
    suppressed = [a for a in analyses if a.coverage_at_fire >= cov_threshold]
    return {
        "description": f"Suppress when coverage >= {cov_threshold} at firing step",
        "suppressed_count": len(suppressed),
        "total": len(analyses),
        "suppressed_ids": [a.trace_id for a in suppressed],
        "remaining": len(analyses) - len(suppressed),
    }


def simulate_option_b(
    entries: list[dict],
    snapshots_cache: dict[str, list[VitalsSnapshot]],
    new_multiplier: float = 3.0,
) -> dict:
    """Option B: raise multiplier. Re-runs detection with new threshold."""
    cfg = VitalsConfig.from_yaml()
    # We need to check which traces still fire with the new multiplier
    from dataclasses import replace as dc_replace

    new_cfg = dc_replace(cfg, burn_rate_multiplier=new_multiplier)
    still_fire = []
    suppressed = []

    for entry in entries:
        tid = entry["trace_id"]
        snaps = snapshots_cache[tid]
        result = analyze_firing_step(snaps, new_cfg)
        if result is not None:
            still_fire.append(tid)
        else:
            suppressed.append(tid)

    return {
        "description": f"Raise multiplier from 2.5 to {new_multiplier}",
        "suppressed_count": len(suppressed),
        "total": len(entries),
        "suppressed_ids": suppressed,
        "still_firing": len(still_fire),
    }


def simulate_option_c(analyses: list[FiringAnalysis], min_depth: int = 3) -> dict:
    """Option C: require minimum baseline depth."""
    suppressed = [a for a in analyses if a.baseline_depth < min_depth]
    return {
        "description": f"Require >= {min_depth} baseline samples",
        "suppressed_count": len(suppressed),
        "total": len(analyses),
        "suppressed_ids": [a.trace_id for a in suppressed],
        "remaining": len(analyses) - len(suppressed),
    }


def main() -> None:
    print("Runaway_cost FP Sensitivity Analysis")
    print("=" * 50)

    cfg = VitalsConfig.from_yaml()
    manifest = load_manifest("v1")

    # Filter to runaway_cost-relevant traces with sufficient confidence
    runaway_entries = [
        e for e in manifest
        if e["labels"].get("runaway_cost") is not None
        and e.get("metadata", {}).get("confidence", 0.0) >= 0.8
    ]
    print(f"Runaway_cost traces in corpus: {len(runaway_entries)}")

    positives = [e for e in runaway_entries if e["labels"]["runaway_cost"] is True]
    negatives = [e for e in runaway_entries if e["labels"]["runaway_cost"] is False]
    print(f"  Labeled positive: {len(positives)}")
    print(f"  Labeled negative: {len(negatives)}")

    # Replay all traces and classify
    print("\nReplaying traces through detector...")
    fp_entries: list[dict] = []
    tp_entries: list[dict] = []
    snapshots_cache: dict[str, list[VitalsSnapshot]] = {}

    for entry in runaway_entries:
        tid = entry["trace_id"]
        snaps = load_trace("v1", entry["path"])
        snapshots_cache[tid] = snaps
        preds = replay_trace(snaps, cfg, runtime_mode="default")

        label = entry["labels"]["runaway_cost"]
        predicted = preds["runaway_cost"]

        if predicted and not label:
            fp_entries.append(entry)
        elif predicted and label:
            tp_entries.append(entry)

    print(f"  TPs: {len(tp_entries)}")
    print(f"  FPs: {len(fp_entries)}")

    # Analyze firing step for all FPs and TPs
    print("\nAnalyzing firing steps...")
    fp_analyses: list[FiringAnalysis] = []
    tp_analyses: list[FiringAnalysis] = []

    for entry in fp_entries:
        analysis = analyze_firing_step(snapshots_cache[entry["trace_id"]], cfg)
        if analysis:
            analysis.trace_id = entry["trace_id"]
            analysis.label = False
            analysis.classification = "FP"
            fp_analyses.append(analysis)

    for entry in tp_entries:
        analysis = analyze_firing_step(snapshots_cache[entry["trace_id"]], cfg)
        if analysis:
            analysis.trace_id = entry["trace_id"]
            analysis.label = True
            analysis.classification = "TP"
            tp_analyses.append(analysis)

    print(f"  FPs with firing analysis: {len(fp_analyses)}")
    print(f"  TPs with firing analysis: {len(tp_analyses)}")

    # === Question 1: Coverage at firing step ===
    fp_cov_above_050 = [a for a in fp_analyses if a.coverage_at_fire >= 0.50]
    fp_cov_above_025 = [a for a in fp_analyses if a.coverage_at_fire >= 0.25]

    # === Question 2: Findings delta at firing step ===
    fp_findings_ge2 = [a for a in fp_analyses if a.findings_delta_at_fire >= 2.0]
    fp_findings_ge1 = [a for a in fp_analyses if a.findings_delta_at_fire >= 1.0]

    # === Question 3: Burn rate ratio distribution ===
    ratio_buckets = {2.5: 0, 3.0: 0, 3.5: 0, 4.0: 0, "4.0+": 0}
    for a in fp_analyses:
        r = a.burn_rate_ratio
        if r <= 2.5:
            ratio_buckets[2.5] += 1
        elif r <= 3.0:
            ratio_buckets[3.0] += 1
        elif r <= 3.5:
            ratio_buckets[3.5] += 1
        elif r <= 4.0:
            ratio_buckets[4.0] += 1
        else:
            ratio_buckets["4.0+"] += 1

    # === Question 4: Safety check — any TP with coverage >= 0.50? ===
    tp_cov_above_050 = [a for a in tp_analyses if a.coverage_at_fire >= 0.50]

    # === Simulate proposed rule changes ===
    print("\nSimulating rule changes...")
    option_a = simulate_option_a(fp_analyses, 0.50)
    option_a_tp_impact = simulate_option_a(tp_analyses, 0.50)

    option_b = simulate_option_b(fp_entries, snapshots_cache, 3.0)
    option_b_tp = simulate_option_b(tp_entries, snapshots_cache, 3.0)

    option_c = simulate_option_c(fp_analyses, 3)
    option_c_tp_impact = simulate_option_c(tp_analyses, 3)

    # Combined A+B
    option_ab_suppressed_fps = set(option_a["suppressed_ids"]) | set(option_b["suppressed_ids"])
    option_ab_suppressed_tps = set(option_a_tp_impact["suppressed_ids"]) | set(
        option_b_tp["suppressed_ids"]
    )

    # === Build reports ===
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_fp = len(fp_analyses)
    total_tp = len(tp_analyses)

    # JSON report
    json_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": "v1",
        "runtime_mode": "default",
        "summary": {
            "total_runaway_traces": len(runaway_entries),
            "labeled_positive": len(positives),
            "labeled_negative": len(negatives),
            "true_positives": len(tp_entries),
            "false_positives": len(fp_entries),
        },
        "question_1_coverage": {
            "description": "FPs with coverage >= 0.50 at firing step",
            "count": len(fp_cov_above_050),
            "total_fps": total_fp,
            "fraction": round(len(fp_cov_above_050) / total_fp, 4) if total_fp else 0,
            "trace_ids": [a.trace_id for a in fp_cov_above_050],
            "also_above_025": {
                "count": len(fp_cov_above_025),
                "fraction": round(len(fp_cov_above_025) / total_fp, 4) if total_fp else 0,
            },
        },
        "question_2_findings_delta": {
            "description": "FPs with findings_delta >= 2 at firing step",
            "count": len(fp_findings_ge2),
            "total_fps": total_fp,
            "fraction": round(len(fp_findings_ge2) / total_fp, 4) if total_fp else 0,
            "trace_ids": [a.trace_id for a in fp_findings_ge2],
            "also_ge_1": {
                "count": len(fp_findings_ge1),
                "fraction": round(len(fp_findings_ge1) / total_fp, 4) if total_fp else 0,
            },
        },
        "question_3_ratio_distribution": {
            "description": "Burn rate ratio distribution at FP firing step (ratio = actual/baseline, detector fires when > 2.5x baseline)",
            "buckets": {str(k): v for k, v in ratio_buckets.items()},
            "total_fps": total_fp,
            "fp_ratios": sorted(
                [{"trace_id": a.trace_id, "ratio": round(a.burn_rate_ratio, 3)} for a in fp_analyses],
                key=lambda x: x["ratio"],
            ),
        },
        "question_4_safety_check": {
            "description": "TPs with coverage >= 0.50 at firing step (safety check for Option A)",
            "count": len(tp_cov_above_050),
            "total_tps": total_tp,
            "safe": len(tp_cov_above_050) == 0,
            "trace_ids": [a.trace_id for a in tp_cov_above_050],
        },
        "option_a": {
            "description": "Coverage suppression: suppress when coverage >= 0.50",
            "fp_suppressed": option_a["suppressed_count"],
            "fp_remaining": option_a["remaining"],
            "tp_suppressed": option_a_tp_impact["suppressed_count"],
            "tp_remaining": option_a_tp_impact["remaining"],
            "safe": option_a_tp_impact["suppressed_count"] == 0,
        },
        "option_b": {
            "description": "Raise multiplier from 2.5 to 3.0",
            "fp_suppressed": option_b["suppressed_count"],
            "fp_remaining": option_b["still_firing"],
            "tp_suppressed": option_b_tp["suppressed_count"],
            "tp_remaining": option_b_tp["still_firing"],
        },
        "option_c": {
            "description": "Minimum baseline depth >= 3",
            "fp_suppressed": option_c["suppressed_count"],
            "fp_remaining": option_c["remaining"],
            "tp_suppressed": option_c_tp_impact["suppressed_count"],
            "tp_remaining": option_c_tp_impact["remaining"],
        },
        "combined_a_b": {
            "description": "Option A + B combined",
            "fp_suppressed": len(option_ab_suppressed_fps),
            "fp_remaining": total_fp - len(option_ab_suppressed_fps),
            "tp_suppressed": len(option_ab_suppressed_tps),
            "tp_remaining": total_tp - len(option_ab_suppressed_tps),
        },
        "fp_details": [
            {
                "trace_id": a.trace_id,
                "firing_step": a.firing_step,
                "total_steps": a.total_steps,
                "coverage_at_fire": round(a.coverage_at_fire, 4),
                "findings_delta_at_fire": round(a.findings_delta_at_fire, 2),
                "token_delta_at_fire": round(a.token_delta_at_fire, 2),
                "burn_rate_ratio": round(a.burn_rate_ratio, 3),
                "baseline_depth": a.baseline_depth,
            }
            for a in sorted(fp_analyses, key=lambda x: x.burn_rate_ratio)
        ],
    }

    # Markdown report
    lines = [
        "# Runaway_cost FP Sensitivity Analysis",
        "",
        f"**Corpus:** v1 | **Mode:** default | **Generated:** {now}",
        "",
        "## Summary",
        "",
        f"- Total runaway_cost traces: {len(runaway_entries)} ({len(positives)} pos, {len(negatives)} neg)",
        f"- True Positives: {len(tp_entries)} | False Positives: {len(fp_entries)}",
        f"- FPs analyzed at firing step: {total_fp} | TPs analyzed: {total_tp}",
        "",
        "---",
        "",
        "## Question 1: Coverage at FP Firing Step",
        "",
        f"**{len(fp_cov_above_050)}/{total_fp}** FPs have coverage >= 0.50 at the firing step "
        f"({len(fp_cov_above_050)/total_fp*100:.1f}%)" if total_fp else "N/A",
        "",
        f"Also: {len(fp_cov_above_025)}/{total_fp} have coverage >= 0.25 "
        f"({len(fp_cov_above_025)/total_fp*100:.1f}%)" if total_fp else "",
        "",
        "| Coverage Bucket | Count | % of FPs |",
        "|----------------|-------|----------|",
    ]

    # Coverage histogram for FPs
    cov_buckets = Counter()
    for a in fp_analyses:
        if a.coverage_at_fire < 0.25:
            cov_buckets["< 0.25"] += 1
        elif a.coverage_at_fire < 0.50:
            cov_buckets["0.25-0.49"] += 1
        elif a.coverage_at_fire < 0.75:
            cov_buckets["0.50-0.74"] += 1
        else:
            cov_buckets[">= 0.75"] += 1

    for bucket in ["< 0.25", "0.25-0.49", "0.50-0.74", ">= 0.75"]:
        count = cov_buckets.get(bucket, 0)
        pct = count / total_fp * 100 if total_fp else 0
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Question 2: Findings Delta at FP Firing Step",
        "",
        f"**{len(fp_findings_ge2)}/{total_fp}** FPs have findings_delta >= 2 at the firing step "
        f"({len(fp_findings_ge2)/total_fp*100:.1f}%)" if total_fp else "N/A",
        "",
        f"Also: {len(fp_findings_ge1)}/{total_fp} have findings_delta >= 1 "
        f"({len(fp_findings_ge1)/total_fp*100:.1f}%)" if total_fp else "",
        "",
        "## Question 3: Burn Rate Ratio Distribution",
        "",
        "Ratio = actual tokens-per-finding / baseline tokens-per-finding at firing step.",
        "Detector fires when ratio > 2.5x baseline (default multiplier).",
        "",
        "| Ratio Bucket | Count | % of FPs |",
        "|-------------|-------|----------|",
    ])

    for bucket_label, bucket_count in ratio_buckets.items():
        pct = bucket_count / total_fp * 100 if total_fp else 0
        lines.append(f"| {bucket_label}x | {bucket_count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Question 4: Safety Check — TPs with coverage >= 0.50",
        "",
        f"**{len(tp_cov_above_050)}/{total_tp}** TPs have coverage >= 0.50 at firing step.",
        "",
    ])
    if len(tp_cov_above_050) == 0:
        lines.append("**SAFE** — Option A (coverage suppression at 0.50) would not suppress any TPs.")
    else:
        lines.append("**UNSAFE** — Option A would suppress the following TPs:")
        for a in tp_cov_above_050:
            lines.append(f"  - {a.trace_id} (cov={a.coverage_at_fire:.3f})")

    lines.extend([
        "",
        "---",
        "",
        "## Proposed Rule Changes — Impact Assessment",
        "",
        "| Option | Description | FPs Suppressed | FPs Remaining | TPs Lost | Safe? |",
        "|--------|-------------|---------------|---------------|----------|-------|",
        f"| A | Coverage >= 0.50 suppression | {option_a['suppressed_count']}/{total_fp} | "
        f"{option_a['remaining']} | {option_a_tp_impact['suppressed_count']}/{total_tp} | "
        f"{'YES' if option_a_tp_impact['suppressed_count'] == 0 else 'NO'} |",
        f"| B | Multiplier 2.5 → 3.0 | {option_b['suppressed_count']}/{total_fp} | "
        f"{option_b['still_firing']} | {option_b_tp['suppressed_count']}/{total_tp} | "
        f"{'YES' if option_b_tp['suppressed_count'] == 0 else 'REVIEW'} |",
        f"| C | Min baseline depth >= 3 | {option_c['suppressed_count']}/{total_fp} | "
        f"{option_c['remaining']} | {option_c_tp_impact['suppressed_count']}/{total_tp} | "
        f"{'YES' if option_c_tp_impact['suppressed_count'] == 0 else 'REVIEW'} |",
        f"| A+B | Combined | {len(option_ab_suppressed_fps)}/{total_fp} | "
        f"{total_fp - len(option_ab_suppressed_fps)} | {len(option_ab_suppressed_tps)}/{total_tp} | "
        f"{'YES' if len(option_ab_suppressed_tps) == 0 else 'REVIEW'} |",
        "",
        "---",
        "",
        "## FP Detail Table",
        "",
        "| Trace ID | Step | Total | Coverage | Findings Δ | Token Δ | Ratio | Baseline Depth |",
        "|----------|------|-------|----------|------------|---------|-------|----------------|",
    ])

    for a in sorted(fp_analyses, key=lambda x: x.burn_rate_ratio):
        lines.append(
            f"| {a.trace_id} | {a.firing_step} | {a.total_steps} "
            f"| {a.coverage_at_fire:.3f} | {a.findings_delta_at_fire:.1f} "
            f"| {a.token_delta_at_fire:.0f} | {a.burn_rate_ratio:.2f}x "
            f"| {a.baseline_depth} |"
        )

    report_md = "\n".join(lines) + "\n"

    # Write reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    md_path = reports_dir / "runaway-fp-sensitivity.md"
    md_path.write_text(report_md)
    print(f"\nMarkdown: {md_path}")

    json_path = reports_dir / "runaway-fp-sensitivity.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    print(f"JSON: {json_path}")

    # Print summary
    print(f"\n{'=' * 50}")
    print("ANSWERS TO AGENT-VITALS QUESTIONS:")
    print(f"{'=' * 50}")
    print(f"\n1. Coverage >= 0.50 at firing step: {len(fp_cov_above_050)}/{total_fp} FPs "
          f"({len(fp_cov_above_050)/total_fp*100:.1f}%)" if total_fp else "")
    print(f"2. Findings_delta >= 2 at firing step: {len(fp_findings_ge2)}/{total_fp} FPs "
          f"({len(fp_findings_ge2)/total_fp*100:.1f}%)" if total_fp else "")
    print(f"3. Ratio distribution: {dict(ratio_buckets)}")
    print(f"4. Safety check (TPs with cov >= 0.50): {len(tp_cov_above_050)}/{total_tp} "
          f"— {'SAFE' if len(tp_cov_above_050) == 0 else 'UNSAFE'}")
    print(f"\nOption A impact: {option_a['suppressed_count']}/{total_fp} FPs removed, "
          f"{option_a_tp_impact['suppressed_count']}/{total_tp} TPs lost")
    print(f"Option B impact: {option_b['suppressed_count']}/{total_fp} FPs removed, "
          f"{option_b_tp['suppressed_count']}/{total_tp} TPs lost")
    print(f"Option C impact: {option_c['suppressed_count']}/{total_fp} FPs removed, "
          f"{option_c_tp_impact['suppressed_count']}/{total_tp} TPs lost")
    print(f"A+B combined: {len(option_ab_suppressed_fps)}/{total_fp} FPs removed, "
          f"{len(option_ab_suppressed_tps)}/{total_tp} TPs lost")


if __name__ == "__main__":
    main()
