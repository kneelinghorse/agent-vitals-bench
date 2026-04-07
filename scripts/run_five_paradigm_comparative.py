"""Five-Paradigm Comparative Report driver (sprint 16, mission s16-m01).

Runs every detection paradigm (handcrafted / causal / TDA / Mamba / Hopfield)
through the M04 partial-trace evaluation harness on the v1 corpus and writes
a paper-ready Markdown + machine-readable JSON report.

Usage
-----

    # From the tda-experiment Python 3.12 venv (has torch + gtda + agent-vitals)
    PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python \
        scripts/run_five_paradigm_comparative.py

    # Or via the Make target which picks the right interpreter
    make five-paradigm-comparative

Outputs
-------

- ``reports/eval-five-paradigm-comparative.md`` — paper-ready report with
  full-trace metric tables, early-detection F1 curves, per-detector winner
  tables, failure-mode discussion, and production deployment guidance.
- ``reports/eval-five-paradigm-comparative.json`` — machine-readable cell
  table that can regenerate every figure in the report without re-running
  the detectors.

Design notes
------------

**Standalone semantics, not hybrid.** Each paradigm scores on its own; we do
not silently fall back to handcrafted when a paradigm cannot answer (the
TDA window-size floor, Mamba/Hopfield ``min_steps``, causal insufficient
history). Each cell reports its own evaluated trace count so coverage gaps
are visible. This is the only framing that makes sense for an early-detection
benchmark — at small cutoffs the learned paradigms genuinely cannot run,
and a hybrid fallback would mask that gap behind handcrafted's predictions.

**Hopfield uses prefix-trained models.** At cutoff 3/5/7 the harness loads
the matching ``_p3 / _p5 / _p7`` Hopfield artifacts; at full trace it loads
the standard model. The other paradigms use a single model across cutoffs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.metrics import DetectorMetrics  # noqa: E402
from evaluator.partial_trace import (  # noqa: E402
    CutoffMetrics,
    PartialTraceConfig,
    _filter_manifest,
    evaluate_at_cutoff,
)
from evaluator.runner import DETECTORS, load_manifest, load_trace  # noqa: E402
from prototypes.predictor_adapters import PARADIGMS, build_predictor  # noqa: E402

DEFAULT_CUTOFFS: tuple[int | None, ...] = (3, 5, 7, None)
DEFAULT_MD_PATH = REPO_ROOT / "reports" / "eval-five-paradigm-comparative.md"
DEFAULT_JSON_PATH = REPO_ROOT / "reports" / "eval-five-paradigm-comparative.json"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """One cell of the (paradigm × cutoff × detector) grid."""

    paradigm: str
    cutoff: int | None
    detector: str
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    precision_lb: float
    recall_lb: float
    evaluated_traces: int

    @property
    def cutoff_label(self) -> str:
        return "full" if self.cutoff is None else f"prefix-{self.cutoff}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "paradigm": self.paradigm,
            "cutoff": self.cutoff,
            "cutoff_label": self.cutoff_label,
            "detector": self.detector,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "precision_lb": round(self.precision_lb, 4),
            "recall_lb": round(self.recall_lb, 4),
            "evaluated_traces": self.evaluated_traces,
        }


@dataclass(frozen=True)
class ComparativeReport:
    corpus_version: str
    min_confidence: float
    paradigms: tuple[str, ...]
    cutoffs: tuple[int | None, ...]
    detectors: tuple[str, ...]
    filtered_trace_count: int
    label_positive_counts: dict[str, int]
    cells: tuple[CellResult, ...]
    timing_seconds: dict[str, float]
    generated_at: str

    def cell(self, *, paradigm: str, cutoff: int | None, detector: str) -> CellResult:
        for c in self.cells:
            if c.paradigm == paradigm and c.cutoff == cutoff and c.detector == detector:
                return c
        raise KeyError(f"no cell for paradigm={paradigm} cutoff={cutoff} detector={detector}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "min_confidence": self.min_confidence,
            "paradigms": list(self.paradigms),
            "cutoffs": [c if c is not None else None for c in self.cutoffs],
            "detectors": list(self.detectors),
            "filtered_trace_count": self.filtered_trace_count,
            "label_positive_counts": self.label_positive_counts,
            "generated_at": self.generated_at,
            "timing_seconds": self.timing_seconds,
            "cells": [c.as_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _cells_from_cutoff_metrics(
    paradigm: str,
    cutoff: int | None,
    metrics: CutoffMetrics,
) -> list[CellResult]:
    cells: list[CellResult] = []
    for detector in DETECTORS:
        m: DetectorMetrics = metrics.detector_metrics[detector]
        p_lb, _ = m.precision_ci
        r_lb, _ = m.recall_ci
        cells.append(
            CellResult(
                paradigm=paradigm,
                cutoff=cutoff,
                detector=detector,
                tp=m.tp,
                fp=m.fp,
                fn=m.fn,
                tn=m.tn,
                precision=m.precision,
                recall=m.recall,
                f1=m.f1,
                precision_lb=p_lb,
                recall_lb=r_lb,
                evaluated_traces=metrics.trace_count,
            )
        )
    return cells


def run_comparative(
    *,
    corpus_version: str = "v1",
    min_confidence: float = 0.8,
    cutoffs: Sequence[int | None] = DEFAULT_CUTOFFS,
    paradigms: Sequence[str] = PARADIGMS,
    verbose: bool = True,
) -> ComparativeReport:
    """Run all paradigms × cutoffs and assemble the comparative report.

    Loads each trace once and reuses it across all (paradigm, cutoff) cells.
    """
    config = PartialTraceConfig(
        cutoffs=tuple(cutoffs),
        min_confidence=min_confidence,
    )
    raw_manifest = load_manifest(corpus_version)
    manifest = _filter_manifest(raw_manifest, config=config)

    if verbose:
        print(
            f"[comparative] loaded manifest: {len(manifest)} entries "
            f"(corpus={corpus_version}, min_confidence={min_confidence})",
            flush=True,
        )

    snapshots_by_path: dict[str, list[Any]] = {}
    for entry in manifest:
        path = str(entry["path"])
        if path not in snapshots_by_path:
            snapshots_by_path[path] = load_trace(corpus_version, path)

    label_positive_counts = {
        detector: sum(1 for entry in manifest if entry["labels"].get(detector, False))
        for detector in DETECTORS
    }

    cells: list[CellResult] = []
    timing: dict[str, float] = {}

    for paradigm in paradigms:
        paradigm_start = time.perf_counter()
        for cutoff in cutoffs:
            cell_start = time.perf_counter()
            predictor = build_predictor(paradigm, cutoff=cutoff)
            cutoff_metrics = evaluate_at_cutoff(
                corpus_version=corpus_version,
                cutoff=cutoff,
                config=config,
                predictor=predictor,
                manifest=manifest,
                snapshots_by_path=snapshots_by_path,
            )
            cell_seconds = time.perf_counter() - cell_start
            new_cells = _cells_from_cutoff_metrics(paradigm, cutoff, cutoff_metrics)
            cells.extend(new_cells)
            label = "full" if cutoff is None else f"prefix-{cutoff}"
            timing[f"{paradigm}.{label}"] = cell_seconds
            if verbose:
                f1_summary = ", ".join(f"{cell.detector}={cell.f1:.3f}" for cell in new_cells)
                print(
                    f"[comparative] {paradigm}@{label}: {cell_seconds:.1f}s  {f1_summary}",
                    flush=True,
                )
        timing[f"{paradigm}.total"] = time.perf_counter() - paradigm_start

    return ComparativeReport(
        corpus_version=corpus_version,
        min_confidence=min_confidence,
        paradigms=tuple(paradigms),
        cutoffs=tuple(cutoffs),
        detectors=tuple(DETECTORS),
        filtered_trace_count=len(manifest),
        label_positive_counts=label_positive_counts,
        cells=tuple(cells),
        timing_seconds=timing,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_cutoff(cutoff: int | None) -> str:
    return "Full" if cutoff is None else f"{cutoff} steps"


def _full_trace_table(
    report: ComparativeReport,
    *,
    metric_attr: str,
    decimals: int = 3,
) -> list[str]:
    """Return markdown rows for a 5×5 (paradigm × detector) full-trace table."""
    headers = ["Paradigm"] + list(report.detectors)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for paradigm in report.paradigms:
        row = [paradigm]
        for detector in report.detectors:
            cell = report.cell(paradigm=paradigm, cutoff=None, detector=detector)
            value = getattr(cell, metric_attr)
            row.append(f"{value:.{decimals}f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _early_detection_table(
    report: ComparativeReport,
    *,
    detector: str,
    metric_attr: str = "f1",
    decimals: int = 3,
) -> list[str]:
    """Return rows for a (paradigm × cutoff) early-detection F1 table."""
    headers = ["Paradigm"] + [_format_cutoff(c) for c in report.cutoffs]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for paradigm in report.paradigms:
        row = [paradigm]
        for cutoff in report.cutoffs:
            cell = report.cell(paradigm=paradigm, cutoff=cutoff, detector=detector)
            value = getattr(cell, metric_attr)
            row.append(f"{value:.{decimals}f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _winner_for(
    report: ComparativeReport,
    *,
    cutoff: int | None,
    detector: str,
) -> CellResult:
    candidates = [
        report.cell(paradigm=p, cutoff=cutoff, detector=detector) for p in report.paradigms
    ]
    return max(
        candidates,
        key=lambda c: (c.f1, c.recall_lb, c.precision_lb),
    )


def _winner_table(report: ComparativeReport, *, cutoff: int | None) -> list[str]:
    headers = ["Detector", "Winner", "F1", "P_lb", "R_lb", "TP", "FP", "FN"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for detector in report.detectors:
        winner = _winner_for(report, cutoff=cutoff, detector=detector)
        lines.append(
            "| "
            + " | ".join(
                [
                    detector,
                    winner.paradigm,
                    f"{winner.f1:.3f}",
                    f"{winner.precision_lb:.3f}",
                    f"{winner.recall_lb:.3f}",
                    str(winner.tp),
                    str(winner.fp),
                    str(winner.fn),
                ]
            )
            + " |"
        )
    return lines


def _macro_f1(report: ComparativeReport, *, paradigm: str, cutoff: int | None) -> float:
    cells = [
        report.cell(paradigm=paradigm, cutoff=cutoff, detector=detector)
        for detector in report.detectors
    ]
    if not cells:
        return 0.0
    return sum(c.f1 for c in cells) / len(cells)


def _macro_table(report: ComparativeReport) -> list[str]:
    headers = ["Paradigm"] + [_format_cutoff(c) for c in report.cutoffs]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for paradigm in report.paradigms:
        row = [paradigm]
        for cutoff in report.cutoffs:
            row.append(f"{_macro_f1(report, paradigm=paradigm, cutoff=cutoff):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _coverage_table(report: ComparativeReport) -> list[str]:
    headers = ["Paradigm"] + [_format_cutoff(c) for c in report.cutoffs]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for paradigm in report.paradigms:
        row = [paradigm]
        for cutoff in report.cutoffs:
            cell = report.cell(paradigm=paradigm, cutoff=cutoff, detector=report.detectors[0])
            row.append(f"{cell.evaluated_traces}/{report.filtered_trace_count}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: ComparativeReport) -> str:
    lines: list[str] = []
    lines.append("# Five-Paradigm Comparative Report")
    lines.append("")
    lines.append(f"**Generated:** {report.generated_at}")
    lines.append(f"**Corpus:** {report.corpus_version}")
    lines.append(f"**Min confidence:** {report.min_confidence:.2f}")
    lines.append(f"**Filtered traces:** {report.filtered_trace_count}")
    lines.append("**Detectors:** " + ", ".join(report.detectors) + " (canonical bench five)")
    lines.append(
        "**Paradigms:** "
        + ", ".join(report.paradigms)
        + " (handcrafted = production rules; causal = rolling causal-window prototype; "
        + "TDA = persistence-feature gradient boosting; Mamba = state-space sequence model; "
        + "Hopfield = modern Hopfield network with learnable stored patterns)"
    )
    lines.append(
        "**Cutoffs:** "
        + ", ".join(_format_cutoff(c) for c in report.cutoffs)
        + " (early-detection prefixes plus the full trace baseline)"
    )
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")

    full_winners = {
        detector: _winner_for(report, cutoff=None, detector=detector)
        for detector in report.detectors
    }
    full_winner_paradigms = sorted({w.paradigm for w in full_winners.values()})
    lines.append(
        "- **Full-trace per-detector winners:** "
        + ", ".join(
            f"{detector} → {w.paradigm} (F1 {w.f1:.3f})" for detector, w in full_winners.items()
        )
    )
    lines.append(
        "- **Distinct winning paradigms (full trace):** " + ", ".join(full_winner_paradigms)
    )
    macro_full = {
        paradigm: _macro_f1(report, paradigm=paradigm, cutoff=None) for paradigm in report.paradigms
    }
    best_macro = max(macro_full.items(), key=lambda kv: kv[1])
    lines.append(
        f"- **Best macro-F1 across detectors (full trace):** {best_macro[0]} ({best_macro[1]:.3f})"
    )
    coverage_loss_3 = [
        paradigm
        for paradigm in report.paradigms
        if report.cell(paradigm=paradigm, cutoff=3, detector=report.detectors[0]).evaluated_traces
        < report.filtered_trace_count
    ]
    if coverage_loss_3:
        lines.append(
            "- **Eligibility gap at 3-step prefix:** "
            + ", ".join(coverage_loss_3)
            + " (paradigms that drop traces at the smallest cutoff)"
        )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Every paradigm is invoked through the bench M04 partial-trace harness "
        "(`evaluator/partial_trace.py:evaluate_at_cutoff`) via a `TracePredictor` "
        "adapter (`prototypes/predictor_adapters.py`). This guarantees identical "
        "input plumbing — the same manifest filtering, the same trace truncation, "
        "the same confusion-matrix accounting — across all 5 paradigms and all 4 "
        "cutoffs."
    )
    lines.append("")
    lines.append(
        "**Standalone semantics, not hybrid.** Each paradigm scores on its own. "
        "When a paradigm cannot answer (TDA window-floor, Mamba/Hopfield "
        "`min_steps`, causal insufficient history), the affected detectors return "
        "`False` rather than falling back to handcrafted. The harness records the "
        "full corpus size at every cell, so eligibility gaps register as recall "
        "drops rather than being hidden behind a hybrid backstop. This is the only "
        "framing that meaningfully distinguishes paradigms at small cutoffs."
    )
    lines.append("")
    lines.append(
        "**Hopfield uses prefix-trained models.** At cutoff 3 / 5 / 7 the harness "
        "loads `{detector}_p3.pt / _p5.pt / _p7.pt`; at full trace it loads "
        "`{detector}.pt`. The other paradigms use a single model across cutoffs."
    )
    lines.append("")
    lines.append(
        "**Corpus.** v1, "
        f"min_confidence={report.min_confidence:.2f}, "
        f"{report.filtered_trace_count} traces. Per-detector positive counts: "
        + ", ".join(
            f"{detector}={count}" for detector, count in report.label_positive_counts.items()
        )
        + "."
    )
    lines.append("")
    lines.append("## Full-Trace Comparison (5 paradigms × 5 detectors)")
    lines.append("")
    lines.append("### F1")
    lines.append("")
    lines.extend(_full_trace_table(report, metric_attr="f1"))
    lines.append("")
    lines.append("### Precision lower bound (Wilson 95%)")
    lines.append("")
    lines.extend(_full_trace_table(report, metric_attr="precision_lb"))
    lines.append("")
    lines.append("### Recall lower bound (Wilson 95%)")
    lines.append("")
    lines.extend(_full_trace_table(report, metric_attr="recall_lb"))
    lines.append("")
    lines.append("## Per-Detector Winners — Full Trace")
    lines.append("")
    lines.extend(_winner_table(report, cutoff=None))
    lines.append("")
    lines.append("## Early-Detection F1 Curves")
    lines.append("")
    lines.append(
        "One table per detector. Rows are paradigms, columns are step-prefix "
        "cutoffs plus the full-trace baseline. Cells are F1 on the entire filtered "
        "corpus at that cutoff under standalone semantics."
    )
    lines.append("")
    for detector in report.detectors:
        lines.append(f"### {detector}")
        lines.append("")
        lines.extend(_early_detection_table(report, detector=detector))
        lines.append("")

    lines.append("## Per-Cutoff Winners")
    lines.append("")
    for cutoff in report.cutoffs:
        lines.append(f"### {_format_cutoff(cutoff)}")
        lines.append("")
        lines.extend(_winner_table(report, cutoff=cutoff))
        lines.append("")

    lines.append("## Macro-F1 Across Detectors")
    lines.append("")
    lines.append(
        "Mean F1 across the five detectors at each cutoff. Useful as a single-"
        "number paradigm summary, but the per-detector tables above are the "
        "authoritative source for paper claims."
    )
    lines.append("")
    lines.extend(_macro_table(report))
    lines.append("")
    lines.append("## Coverage by Cutoff")
    lines.append("")
    lines.append(
        "Number of traces evaluated per (paradigm, cutoff) cell. The harness "
        "uses `skip_short_traces=False`, so traces shorter than a cutoff are "
        "passed through at their full length rather than skipped — meaning every "
        "paradigm sees the full filtered corpus at every cutoff. This table is "
        "therefore informational; eligibility gaps surface in the F1 / recall "
        "tables above as drops on traces the paradigm can't process."
    )
    lines.append("")
    lines.extend(_coverage_table(report))
    lines.append("")

    lines.append("## Failure Mode Analysis")
    lines.append("")
    lines.append(
        "For each detector, identifies which paradigm holds the lead at each "
        "cutoff. Where the leader changes between cutoffs, that's a signal that "
        "different paradigms shine at different stages of trace progression — "
        "discussion fodder for the eventual paper."
    )
    lines.append("")
    for detector in report.detectors:
        lines.append(f"### {detector}")
        per_cutoff_winner = {
            cutoff: _winner_for(report, cutoff=cutoff, detector=detector)
            for cutoff in report.cutoffs
        }
        unique_winners = sorted({w.paradigm for w in per_cutoff_winner.values()})
        for cutoff in report.cutoffs:
            w = per_cutoff_winner[cutoff]
            lines.append(
                f"- **{_format_cutoff(cutoff)}**: {w.paradigm} (F1 {w.f1:.3f}, "
                f"P_lb {w.precision_lb:.3f}, R_lb {w.recall_lb:.3f}, "
                f"TP={w.tp} FP={w.fp} FN={w.fn})"
            )
        if len(unique_winners) > 1:
            lines.append(
                f"- **Cutoff-dependent leader**: {' / '.join(unique_winners)} — "
                "different paradigms hold the lead at different stages."
            )
        else:
            lines.append(f"- **Stable leader**: {unique_winners[0]} dominates across all cutoffs.")
        lines.append("")

    lines.append("## Production Deployment Guidance")
    lines.append("")
    lines.append(
        "Per-detector recommendations derived programmatically from the per-"
        "detector winner tables. These are bench data; final integration choices "
        "in agent-vitals should weigh them against compute cost, dependency "
        "footprint, and the production framing already in v1.13.x (handcrafted "
        "rules + causal + TDA hybrid override-only)."
    )
    lines.append("")
    lines.append("| Detector | Production Pick | Why |")
    lines.append("|---|---|---|")
    for detector in report.detectors:
        full_winner = full_winners[detector]
        early_winner = _winner_for(report, cutoff=3, detector=detector)
        if full_winner.paradigm == early_winner.paradigm:
            why = (
                f"{full_winner.paradigm} leads both full-trace (F1 {full_winner.f1:.3f}) "
                f"and early-detection (F1 @3 {early_winner.f1:.3f})"
            )
            pick = full_winner.paradigm
        else:
            why = (
                f"{full_winner.paradigm} leads full-trace (F1 {full_winner.f1:.3f}); "
                f"consider {early_winner.paradigm} as an early-detection adjudicator "
                f"(F1 @3 {early_winner.f1:.3f})"
            )
            pick = f"{full_winner.paradigm} (+{early_winner.paradigm} early)"
        lines.append(f"| {detector} | {pick} | {why} |")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Run the report end-to-end from the sibling tda-experiment Python 3.12 "
        "venv (which has torch, gtda, hflayers, agent-vitals editable):"
    )
    lines.append("")
    lines.append("```")
    lines.append("make five-paradigm-comparative")
    lines.append("```")
    lines.append("")
    lines.append("Or directly:")
    lines.append("")
    lines.append("```")
    lines.append(
        "PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python "
        "scripts/run_five_paradigm_comparative.py"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Every figure in this report is regenerable from "
        "`reports/eval-five-paradigm-comparative.json` (paradigm × cutoff × "
        "detector cell list with TP/FP/FN/TN, P/R/F1, P_lb/R_lb, and per-cell "
        "evaluated trace count) without re-running the detectors."
    )
    lines.append("")
    lines.append("### Per-paradigm runtime")
    lines.append("")
    lines.append("| Paradigm | Total seconds |")
    lines.append("|---|---|")
    for paradigm in report.paradigms:
        total = report.timing_seconds.get(f"{paradigm}.total", 0.0)
        lines.append(f"| {paradigm} | {total:.1f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cutoff(value: str) -> int | None:
    if value.lower() in {"full", "none", "null"}:
        return None
    return int(value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="v1", help="Corpus version to evaluate")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum manifest confidence to include",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        default=["3", "5", "7", "full"],
        help="Step-prefix cutoffs to evaluate (use 'full' for the full trace)",
    )
    parser.add_argument(
        "--paradigms",
        nargs="+",
        default=list(PARADIGMS),
        help="Paradigms to evaluate (default: all five)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Markdown report output path",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="JSON report output path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell progress output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cutoffs = tuple(_parse_cutoff(c) for c in args.cutoffs)
    paradigms = tuple(args.paradigms)
    for paradigm in paradigms:
        if paradigm not in PARADIGMS:
            parser.error(f"unknown paradigm {paradigm!r}; expected one of {PARADIGMS}")

    report = run_comparative(
        corpus_version=args.corpus,
        min_confidence=args.min_confidence,
        cutoffs=cutoffs,
        paradigms=paradigms,
        verbose=not args.quiet,
    )

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report) + "\n")
    args.json_output.write_text(json.dumps(report.as_dict(), indent=2) + "\n")

    print(
        f"\n[comparative] wrote {args.markdown_output} and {args.json_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
