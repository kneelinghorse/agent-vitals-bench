#!/usr/bin/env python3
"""Run the causal-aware confabulation elicitation campaign."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.runner import load_trace, replay_trace, resolve_workflow_type
from elicitation.elicit_confabulation_causal import elicit_confabulation_causal
from elicitation.providers import get_provider
from prototypes.causal_confab import detect_causal_confabulation


CORPUS_DIR = PROJECT_ROOT / "corpus" / "v1"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
REPORT_MD = PROJECT_ROOT / "reports" / "eval-causal-confab-elicited.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "eval-causal-confab-elicited.json"
SLICE_PREFIX = "traces/confabulation/elicited_causal/"

CAMPAIGN_MODELS: tuple[dict[str, str], ...] = (
    {
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini-2025-04-14",
        "label": "gpt-4.1-mini",
    },
    {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
        "label": "gemini-2.5-flash",
    },
    {
        "provider": "openrouter",
        "model": "anthropic/claude-4-sonnet-20250522",
        "label": "claude-4-sonnet",
    },
)


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _write_trace(run: Any) -> str:
    label = "positive" if run.metadata.labels["confabulation"] else "negative"
    trace_dir = CORPUS_DIR / "traces" / "confabulation" / "elicited_causal" / label
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{run.trace_id}.json"
    payload = [snapshot.model_dump(mode="json") for snapshot in run.snapshots]
    trace_path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(trace_path.relative_to(CORPUS_DIR))


def _build_manifest_entry(run: Any, rel_path: str, model_label: str) -> dict[str, Any]:
    entry: dict[str, Any] = dict(run.metadata.as_manifest_entry(rel_path))
    entry["metadata"]["model"] = model_label
    entry["metadata"]["provider"] = run.provider_name
    entry["metadata"]["notes"] = (
        f"causal-aware confab campaign; supported_ratio="
        f"{run.metadata.params['final_supported_ratio']}"
    )
    return entry


def _save_manifest(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loaded = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    manifest = loaded if isinstance(loaded, list) else []
    existing_ids = {entry["trace_id"] for entry in manifest}
    for entry in entries:
        if entry["trace_id"] not in existing_ids:
            manifest.append(entry)
            existing_ids.add(entry["trace_id"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _evaluate_slice(entries: list[dict[str, Any]], min_confidence: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        confidence = float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        snapshots = load_trace("v1", str(entry["path"]))
        mission_id = snapshots[0].mission_id if snapshots else None
        workflow_type = resolve_workflow_type(str(entry["trace_id"]), mission_id)
        handcrafted = replay_trace(snapshots, workflow_type=workflow_type)["confabulation"]
        causal = detect_causal_confabulation(snapshots).detected
        rows.append(
            {
                "trace_id": entry["trace_id"],
                "path": entry["path"],
                "model": entry.get("metadata", {}).get("model"),
                "provider": entry.get("metadata", {}).get("provider"),
                "label": bool(entry["labels"].get("confabulation", False)),
                "confidence": confidence,
                "handcrafted": handcrafted,
                "causal": causal,
            }
        )

    per_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row["model"])
        stats = per_model.setdefault(
            model,
            {
                "provider": row["provider"],
                "positives": 0,
                "negatives": 0,
                "causal_hits": 0,
                "handcrafted_hits": 0,
                "causal_fp": 0,
                "handcrafted_fp": 0,
            },
        )
        if row["label"]:
            stats["positives"] += 1
            stats["causal_hits"] += 1 if row["causal"] else 0
            stats["handcrafted_hits"] += 1 if row["handcrafted"] else 0
        else:
            stats["negatives"] += 1
            stats["causal_fp"] += 1 if row["causal"] else 0
            stats["handcrafted_fp"] += 1 if row["handcrafted"] else 0

    success_positives = sum(stats["positives"] for stats in per_model.values())
    qualifying_models = sum(1 for stats in per_model.values() if stats["positives"] > 0)
    return {
        "min_confidence": min_confidence,
        "trace_count": len(rows),
        "success_positives": success_positives,
        "qualifying_models": qualifying_models,
        "per_model": per_model,
        "rows": rows,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Causal-Aware Confabulation Elicitation Report")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Each step asks for discrete findings, and each finding must provide one DOI.")
    lines.append(
        "- `findings_count` tracks total claims while `sources_count` tracks cumulative distinct "
        "verified supporting DOIs."
    )
    lines.append(
        "- Positive runs use hyper-niche recent topics and forbid DOI reuse across steps; negative "
        "runs use broad topics."
    )
    lines.append(
        "- This makes the causal invariant explicit: real supported findings keep sources coupled, "
        "while fabricated support lets findings continue after verified sources stall."
    )
    lines.append("")
    lines.append("## Success Criteria")
    lines.append("")
    lines.append(f"- High-confidence positive traces: {summary['success_positives']} (target: 15)")
    lines.append(
        f"- Model families with high-confidence positives: {summary['qualifying_models']} "
        f"(target: 3)"
    )
    lines.append(f"- Total evaluated traces: {summary['trace_count']}")
    lines.append("")
    lines.append("## Per-Model Detection Rates")
    lines.append("")
    lines.append(
        "| Model | Provider | Positives | Negatives | Causal Rate | Handcrafted Rate | "
        "Causal FP | Handcrafted FP |"
    )
    lines.append(
        "|-------|----------|-----------|-----------|-------------|------------------|-----------|----------------|"
    )
    for model, stats in sorted(summary["per_model"].items()):
        positives = int(stats["positives"])
        negatives = int(stats["negatives"])
        causal_rate = stats["causal_hits"] / positives if positives else 0.0
        handcrafted_rate = stats["handcrafted_hits"] / positives if positives else 0.0
        lines.append(
            f"| {model} | {stats['provider']} | {positives} | {negatives} | "
            f"{causal_rate:.3f} | {handcrafted_rate:.3f} | {stats['causal_fp']} | "
            f"{stats['handcrafted_fp']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_report(summary: dict[str, Any]) -> None:
    REPORT_MD.write_text(_render_report(summary) + "\n")
    REPORT_JSON.write_text(json.dumps(summary, indent=2) + "\n")


def _collect_for_model(
    model_cfg: dict[str, str],
    *,
    positive_target: int,
    negative_target: int,
    max_attempts: int,
    min_confidence: float,
    findings_per_step: int,
    total_steps: int,
    api_key: str,
) -> list[dict[str, Any]]:
    provider = get_provider(model_cfg["provider"], model=model_cfg["model"])
    entries: list[dict[str, Any]] = []

    pos_hits = 0
    pos_attempts = 0
    print(f"Collecting causal confab traces for {model_cfg['label']} via {model_cfg['provider']}")
    while pos_hits < positive_target and pos_attempts < max_attempts:
        pos_attempts += 1
        try:
            run = elicit_confabulation_causal(
                provider,
                positive=True,
                total_steps=total_steps,
                findings_per_step=findings_per_step,
                verify_api_key=api_key,
            )
        except Exception as exc:
            print(f"  positive attempt {pos_attempts}: error={exc!r}")
            continue
        if run.metadata.confidence < min_confidence:
            print(
                f"  positive attempt {pos_attempts}: low confidence {run.metadata.confidence:.2f}"
            )
            continue
        rel_path = _write_trace(run)
        entries.append(_build_manifest_entry(run, rel_path, model_cfg["label"]))
        if run.metadata.labels["confabulation"]:
            pos_hits += 1
            print(
                f"  positive attempt {pos_attempts}: kept as positive ({pos_hits}/{positive_target})"
            )
        else:
            print(
                "  positive attempt {pos_attempts}: saved but labeled negative".format(
                    pos_attempts=pos_attempts
                )
            )

    neg_hits = 0
    neg_attempts = 0
    while neg_hits < negative_target and neg_attempts < max_attempts:
        neg_attempts += 1
        try:
            run = elicit_confabulation_causal(
                provider,
                positive=False,
                total_steps=total_steps,
                findings_per_step=findings_per_step,
                verify_api_key=api_key,
            )
        except Exception as exc:
            print(f"  negative attempt {neg_attempts}: error={exc!r}")
            continue
        if run.metadata.confidence < min_confidence:
            print(
                f"  negative attempt {neg_attempts}: low confidence {run.metadata.confidence:.2f}"
            )
            continue
        rel_path = _write_trace(run)
        entries.append(_build_manifest_entry(run, rel_path, model_cfg["label"]))
        if not run.metadata.labels["confabulation"]:
            neg_hits += 1
            print(
                f"  negative attempt {neg_attempts}: kept as negative ({neg_hits}/{negative_target})"
            )
        else:
            print(
                "  negative attempt {neg_attempts}: saved but labeled positive".format(
                    neg_attempts=neg_attempts
                )
            )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-target", type=int, default=5)
    parser.add_argument("--negative-target", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--findings-per-step", type=int, default=10)
    parser.add_argument("--total-steps", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _load_env()
    if args.dry_run:
        print("Causal confab campaign plan")
        for model in CAMPAIGN_MODELS:
            print(
                f"  {model['label']} via {model['provider']}: "
                f"{args.positive_target} positives, {args.negative_target} negatives"
            )
        return

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    new_entries: list[dict[str, Any]] = []
    for model_cfg in CAMPAIGN_MODELS:
        model_entries = _collect_for_model(
            model_cfg,
            positive_target=args.positive_target,
            negative_target=args.negative_target,
            max_attempts=args.max_attempts,
            min_confidence=args.min_confidence,
            findings_per_step=args.findings_per_step,
            total_steps=args.total_steps,
            api_key=api_key,
        )
        new_entries.extend(model_entries)

    manifest = _save_manifest(new_entries)
    slice_entries = [entry for entry in manifest if str(entry["path"]).startswith(SLICE_PREFIX)]
    summary = _evaluate_slice(slice_entries, args.min_confidence)
    _write_report(summary)

    print(_render_report(summary))
    print(f"\nManifest updated: {MANIFEST_PATH}")
    print(f"Markdown report: {REPORT_MD}")
    print(f"JSON report: {REPORT_JSON}")


if __name__ == "__main__":
    main()
