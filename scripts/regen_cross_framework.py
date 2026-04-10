"""Cross-framework gate regen — explicit default vs tda runtime modes.

Runs the canonical bench evaluator across the full {default, crewai, dspy,
langgraph} × {default, tda} grid (8 cells) and emits one combined Markdown
report plus one combined JSON cell payload that the metrics_and_protocols
team can rebuild Sprint 15 paper tables from without any ambiguity.

This is the post-replay-audit successor to the pre-audit per-framework
reports under reports/2026-04-07T14:45*. Both runtime modes use the
canonical agent_vitals.backtest._replay_trace via bench's repaired
evaluator/runner.replay_trace, so the numbers in this report are mode-
explicit and drift-free.

Usage
-----
    PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python \
        scripts/regen_cross_framework.py

The tda-experiment Python 3.12 venv has the TDA backend installed, which is
required for the tda runtime-mode cells. The bench main venv has tda
unavailable and would silently produce incorrect tda-mode numbers.

Outputs
-------
- reports/eval-cross-framework-v1.md   — paper-ready combined report
- reports/eval-cross-framework-v1.json — machine-readable cell payload
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_vitals.detection.tda import is_tda_available  # noqa: E402

from evaluator.runner import DETECTORS, EvaluationResult, run_evaluation  # noqa: E402

# Profile slug used in this report for the no-profile case (i.e. agent-vitals
# default thresholds with no framework override applied).
DEFAULT_PROFILE_LABEL = "default"
PROFILES: tuple[str, ...] = (DEFAULT_PROFILE_LABEL, "crewai", "dspy", "langgraph")
RUNTIME_MODES: tuple[str, ...] = ("default", "tda")

DEFAULT_MD_PATH = REPO_ROOT / "reports" / "eval-cross-framework-v1.md"
DEFAULT_JSON_PATH = REPO_ROOT / "reports" / "eval-cross-framework-v1.json"


@dataclass(frozen=True)
class CellSummary:
    """One (profile × runtime_mode) evaluation cell."""

    profile: str
    runtime_mode: str
    trace_count: int
    detector_metrics: dict[str, dict[str, Any]]
    gate_results: dict[str, dict[str, Any]]
    excluded_detectors: tuple[str, ...]
    composite_passed: bool
    composite_pass_count: int
    composite_total: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "runtime_mode": self.runtime_mode,
            "trace_count": self.trace_count,
            "detector_metrics": self.detector_metrics,
            "gate_results": self.gate_results,
            "excluded_detectors": list(self.excluded_detectors),
            "composite_passed": self.composite_passed,
            "composite_pass_count": self.composite_pass_count,
            "composite_total": self.composite_total,
        }


def _summarize_cell(result: EvaluationResult) -> CellSummary:
    detector_metrics = {name: m.as_dict() for name, m in result.detector_metrics.items()}
    gate_results = result.gate_results
    excluded = set(result.excluded_detectors)
    non_excluded = [(name, gate) for name, gate in gate_results.items() if name not in excluded]
    pass_count = sum(1 for _, g in non_excluded if g.get("passed", False))
    total = len(non_excluded)
    composite_passed = pass_count == total and total > 0
    return CellSummary(
        profile=result.profile or DEFAULT_PROFILE_LABEL,
        runtime_mode=result.runtime_mode,
        trace_count=result.trace_count,
        detector_metrics=detector_metrics,
        gate_results=gate_results,
        excluded_detectors=tuple(sorted(excluded)),
        composite_passed=composite_passed,
        composite_pass_count=pass_count,
        composite_total=total,
    )


def run_grid() -> list[CellSummary]:
    if not is_tda_available():
        raise RuntimeError(
            "TDA backend not available in current interpreter — run from the "
            "tda-experiment Python 3.12 venv (PYTHONPATH=<bench> "
            "../tda-experiment/.venv/bin/python scripts/regen_cross_framework.py). "
            "The bench main venv does not ship the TDA backend and would produce "
            "incorrect tda-mode numbers."
        )

    cells: list[CellSummary] = []
    for profile in PROFILES:
        for mode in RUNTIME_MODES:
            profile_arg = None if profile == DEFAULT_PROFILE_LABEL else profile
            print(
                f"[cross-framework] running profile={profile} runtime_mode={mode}",
                flush=True,
            )
            result = run_evaluation(
                corpus_version="v1",
                profile=profile_arg,
                runtime_mode=mode,
            )
            cell = _summarize_cell(result)
            cells.append(cell)
            print(
                f"[cross-framework]   trace_count={cell.trace_count} "
                f"composite={cell.composite_pass_count}/{cell.composite_total} "
                f"({'PASS' if cell.composite_passed else 'NO-GO'})",
                flush=True,
            )
    return cells


def _gate_status_label(gate: dict[str, Any]) -> str:
    if gate.get("excluded"):
        return "EXCLUDED"
    return "HARD GATE" if gate.get("passed") else "NO-GO"


def _format_pct(value: float) -> str:
    return f"{value:.4f}"


def _per_detector_table(cell: CellSummary) -> list[str]:
    lines = [
        "| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for detector in DETECTORS:
        m = cell.detector_metrics[detector]
        gate = cell.gate_results[detector]
        status = _gate_status_label(gate)
        lines.append(
            "| "
            + " | ".join(
                [
                    detector,
                    status,
                    str(m["tp"]),
                    str(m["fp"]),
                    str(m["fn"]),
                    str(m["tn"]),
                    _format_pct(m["precision"]),
                    _format_pct(m["recall"]),
                    _format_pct(m["f1"]),
                    _format_pct(m["precision_lb"]),
                    _format_pct(m["recall_lb"]),
                ]
            )
            + " |"
        )
    return lines


def _profile_section(profile: str, cells_by_mode: dict[str, CellSummary]) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Profile: {profile}")
    lines.append("")
    headline_parts: list[str] = []
    for mode in RUNTIME_MODES:
        cell = cells_by_mode[mode]
        composite = (
            f"{cell.composite_pass_count}/{cell.composite_total} "
            f"({'PASS' if cell.composite_passed else 'NO-GO'})"
        )
        excluded_note = (
            f" — excluded: {', '.join(cell.excluded_detectors)}" if cell.excluded_detectors else ""
        )
        headline_parts.append(
            f"**{mode} runtime mode** — composite {composite}, "
            f"trace_count={cell.trace_count}{excluded_note}"
        )
    lines.append("- " + "\n- ".join(headline_parts))
    lines.append("")
    for mode in RUNTIME_MODES:
        cell = cells_by_mode[mode]
        lines.append(f"### {profile} × {mode} runtime mode")
        lines.append("")
        lines.extend(_per_detector_table(cell))
        lines.append("")
        if cell.excluded_detectors:
            lines.append(
                "*Excluded from composite gate (disabled by profile): "
                + ", ".join(cell.excluded_detectors)
                + ".*"
            )
            lines.append("")
    return lines


def _composite_grid_table(cells: list[CellSummary]) -> list[str]:
    by_pair = {(c.profile, c.runtime_mode): c for c in cells}
    headers = ["Profile"] + [f"{mode} mode" for mode in RUNTIME_MODES]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for profile in PROFILES:
        row = [profile]
        for mode in RUNTIME_MODES:
            cell = by_pair[(profile, mode)]
            label = "PASS" if cell.composite_passed else "NO-GO"
            row.append(f"{cell.composite_pass_count}/{cell.composite_total} {label}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _runaway_lb_grid(cells: list[CellSummary]) -> list[str]:
    """Cross-cut on runaway_cost P_lb specifically — the load-bearing number
    for the publication HARD GATE discussion in m02 §8.4 / m03 §4.3."""
    by_pair = {(c.profile, c.runtime_mode): c for c in cells}
    headers = ["Profile", "default mode P_lb", "tda mode P_lb"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for profile in PROFILES:
        row = [profile]
        for mode in RUNTIME_MODES:
            cell = by_pair[(profile, mode)]
            gate = cell.gate_results["runaway_cost"]
            row.append(f"{gate['precision_lb']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(cells: list[CellSummary], *, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Cross-Framework Gate Report — Post-Replay-Audit")
    lines.append("")
    lines.append(f"**Generated:** {generated_at}")
    lines.append("**Corpus:** v1, min_confidence ≥ 0.80")
    lines.append(
        "**Detectors:** loop, stuck, confabulation, thrash, runaway_cost (canonical bench five)"
    )
    lines.append(
        "**Profiles:** default, crewai, dspy, langgraph "
        "(default = no framework override; the other three apply the "
        "agent-vitals thresholds.yaml profile via the v1.14.0 public "
        "VitalsConfig profile-introspection API)"
    )
    lines.append(
        "**Runtime modes:** default (`tda_enabled=False`, pure handcrafted "
        "rules) and tda (`tda_enabled=True`, handcrafted with TDA persistence-"
        "feature gradient-boosting override on runaway_cost only)"
    )
    lines.append(
        "**Replay backend:** canonical `agent_vitals.backtest._replay_trace` "
        "via bench's repaired `evaluator/runner.replay_trace`. No mirror, no "
        "drift, no per-script ad-hoc replay paths."
    )
    lines.append("")
    lines.append("## TL;DR — Composite gate grid")
    lines.append("")
    lines.append(
        "Counts are non-excluded detectors only. DSPy intentionally disables "
        "`workflow_stuck_enabled`, so its `stuck` detector is excluded from "
        "the composite gate (documented framework-specific integration "
        "constraint, not a detector failure)."
    )
    lines.append("")
    lines.extend(_composite_grid_table(cells))
    lines.append("")
    lines.append("## TL;DR — runaway_cost precision lower bound")
    lines.append("")
    lines.append(
        "Cross-cut on the load-bearing number for the m02 §8.4 / m03 §4.3 "
        "discussion. Publication HARD GATE threshold is P_lb ≥ 0.80."
    )
    lines.append("")
    lines.extend(_runaway_lb_grid(cells))
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Each (profile, runtime_mode) cell is one full pass through "
        "`evaluator.runner.run_evaluation` on corpus v1 with "
        "`min_confidence=0.80`. Profile selection threads through "
        "`VitalsConfig.for_framework(profile)` exactly as production does. "
        "Runtime mode is enforced explicitly: `default` forces "
        "`tda_enabled=False`, `tda` forces `tda_enabled=True`. The trace "
        "loop and confusion-matrix accounting are identical across all 8 "
        "cells — only the config differs."
    )
    lines.append("")
    lines.append(
        "Manifest filtering also threads the profile: when a profile is "
        "active, traces tagged with a *different* framework are excluded, "
        "but untagged traces are kept. This means the per-profile trace "
        "counts may differ slightly from the default-profile baseline; the "
        "trace_count line in each section reports the actual count."
    )
    lines.append("")
    lines.append(
        "**Replay backend.** Both runtime modes go through "
        "`agent_vitals.backtest._replay_trace` via bench's repaired "
        "`evaluator/runner.replay_trace`. This is the same canonical "
        "production replay path that the v1.14.0 acceptance and the "
        "regenerated five-paradigm comparative report both rely on. The "
        "pre-audit per-framework reports under `reports/2026-04-07T14:45*` "
        "predate the replay repair and should be considered superseded by "
        "this report for any paper claim."
    )
    lines.append("")
    lines.append(
        "**Gate thresholds.** From `evaluator/gate.py`: `MIN_POSITIVES=25`, "
        "`MIN_PRECISION_LB=0.80`, `MIN_RECALL_LB=0.75` (Wilson 95% lower "
        "bounds, two-sided). A detector passes if it has enough positives "
        "*and* both lower bounds clear thresholds. The composite gate is "
        "the AND of all non-excluded detectors."
    )
    lines.append("")
    lines.append("## Per-profile detail")
    lines.append("")
    by_profile_mode: dict[str, dict[str, CellSummary]] = {}
    for cell in cells:
        by_profile_mode.setdefault(cell.profile, {})[cell.runtime_mode] = cell
    for profile in PROFILES:
        lines.extend(_profile_section(profile, by_profile_mode[profile]))
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```")
    lines.append(
        "PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python scripts/regen_cross_framework.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Both modes require the TDA backend (the default-mode cells don't "
        "use it, but the script verifies availability at startup so all 8 "
        "cells run from the same interpreter and the report is internally "
        "consistent). Use the tda-experiment Python 3.12 venv. The "
        "machine-readable cell payload is at "
        "`reports/eval-cross-framework-v1.json` — every figure in this "
        "report is regenerable from it without re-running the detectors."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    cells = run_grid()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    DEFAULT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MD_PATH.write_text(render_markdown(cells, generated_at=generated_at) + "\n")
    payload = {
        "generated_at": generated_at,
        "corpus_version": "v1",
        "min_confidence": 0.80,
        "profiles": list(PROFILES),
        "runtime_modes": list(RUNTIME_MODES),
        "detectors": list(DETECTORS),
        "cells": [c.as_dict() for c in cells],
    }
    DEFAULT_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[cross-framework] wrote {DEFAULT_MD_PATH}")
    print(f"[cross-framework] wrote {DEFAULT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
