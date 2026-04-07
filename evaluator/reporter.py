"""Report generation — console, markdown, and JSON output."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.runner import EvaluationResult

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def report_results(result: EvaluationResult, *, save: bool = True) -> str:
    """Generate markdown report from evaluation results.

    Returns the report as a string and optionally saves to reports/ directory.
    """
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    profile_label = result.profile or "default"
    lines.append("# Agent Vitals Bench — Gate Report")
    lines.append("")
    lines.append(f"**Corpus:** {result.corpus_version}")
    lines.append(f"**Profile:** {profile_label}")
    lines.append(f"**Traces evaluated:** {result.trace_count}")
    lines.append(f"**Detectors:** {', '.join(result.detectors_evaluated)}")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # Summary table
    lines.append("## Gate Results")
    lines.append("")
    lines.append("| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |")
    lines.append("|----------|---|---|----|----- |------|-----------|--------|")

    all_passed = True
    excluded_names: list[str] = []
    for name in result.detectors_evaluated:
        gate = result.gate_results.get(name)
        if not gate:
            continue
        if gate.get("excluded", False):
            excluded_names.append(name)
            reason = gate.get("exclude_reason", "disabled by profile")
            lines.append(
                f"| {name} "
                f"| — | — | — | — | — "
                f"| {gate['total_positives']} "
                f"| **EXCLUDED** ({reason}) |"
            )
            continue
        status_icon = "HARD GATE" if gate["passed"] else "NO-GO"
        if not gate["passed"]:
            all_passed = False
        lines.append(
            f"| {name} "
            f"| {gate['precision']:.3f} "
            f"| {gate['recall']:.3f} "
            f"| {gate['f1']:.3f} "
            f"| {gate['precision_lb']:.3f} "
            f"| {gate['recall_lb']:.3f} "
            f"| {gate['total_positives']} "
            f"| **{status_icon}** |"
        )

    lines.append("")
    composite_status = "PASS" if all_passed else "FAIL"
    evaluated_count = len(result.detectors_evaluated) - len(excluded_names)
    if excluded_names:
        lines.append(
            f"**Composite gate:** {composite_status} "
            f"({evaluated_count} evaluated, {len(excluded_names)} excluded: "
            f"{', '.join(excluded_names)})"
        )
    else:
        lines.append(f"**Composite gate:** {composite_status}")
    lines.append("")

    # Detailed confusion matrices
    lines.append("## Confusion Matrices")
    lines.append("")
    for name in result.detectors_evaluated:
        metrics = result.detector_metrics.get(name)
        if not metrics:
            continue
        lines.append(f"### {name}")
        lines.append(f"- TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}, TN={metrics.tn}")
        p_lb, p_ub = metrics.precision_ci
        r_lb, r_ub = metrics.recall_ci
        lines.append(f"- Precision CI: [{p_lb:.3f}, {p_ub:.3f}]")
        lines.append(f"- Recall CI: [{r_lb:.3f}, {r_ub:.3f}]")
        lines.append("")

    # Gate check details
    lines.append("## Gate Check Details")
    lines.append("")
    for name in result.detectors_evaluated:
        gate = result.gate_results.get(name)
        if not gate:
            continue
        lines.append(f"### {name}")
        for check_name, check in gate["checks"].items():
            icon = "pass" if check["passed"] else "FAIL"
            lines.append(
                f"- {check_name}: {check['actual']} (required: {check['required']}) — {icon}"
            )
        lines.append("")

    report_text = "\n".join(lines)

    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        suffix = f"-{result.profile}" if result.profile else ""
        report_path = REPORTS_DIR / f"eval-{date_str}-{result.corpus_version}{suffix}.md"
        report_path.write_text(report_text)

        # Also save JSON for machine consumption
        json_path = REPORTS_DIR / f"eval-{date_str}-{result.corpus_version}{suffix}.json"
        json_data = {
            "corpus_version": result.corpus_version,
            "profile": profile_label,
            "trace_count": result.trace_count,
            "generated_at": now,
            "excluded_detectors": list(result.excluded_detectors),
            "detectors": {
                name: result.detector_metrics[name].as_dict()
                for name in result.detectors_evaluated
                if name in result.detector_metrics
            },
            "gates": result.gate_results,
        }
        json_path.write_text(json.dumps(json_data, indent=2))

    return report_text
