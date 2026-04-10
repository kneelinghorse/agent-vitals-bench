#!/usr/bin/env python3
"""Cross-model elicitation campaign for Sprint 9 validation.

Organizes elicitation by model tier (frontier, mid-range, volume) to validate
detector generalization across model classes. Outputs to corpus/v1/elicited_cross_model/
and updates manifest.json with model_tier metadata.

Tiers:
  frontier  — claude-sonnet, gpt-4o (high capability, expensive)
  mid-range — qwen3.5-72b via OpenRouter, deepseek-chat (strong, cheap)
  volume    — gpt-4o-mini, gemini-2.0-flash (fast, cheapest)

Usage:
    python scripts/elicit_cross_model.py \
        --tiers frontier volume \
        --detectors confabulation thrash runaway_cost \
        --traces-per-tier 20 \
        --negative-ratio 0.25

    # Dry run to see what would be generated
    python scripts/elicit_cross_model.py --dry-run

    # Single tier, single detector
    python scripts/elicit_cross_model.py --tiers volume --detectors confabulation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from agent_vitals.config import VitalsConfig
from agent_vitals.detection.loop import detect_loop
from agent_vitals.detection.stop_rule import derive_stop_signals

from elicitation.providers import get_provider

CORPUS_DIR = PROJECT_ROOT / "corpus" / "v1"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

# Model tier definitions — each tier maps to provider configs.
# Provider routing: DeepSeek direct, MiniMax direct, everything else OpenRouter.
# Check https://openrouter.ai/rankings for current model IDs before updating.
# IMPORTANT: Never run API elicitation without user approval of this model list.
MODEL_TIERS: dict[str, list[dict[str, str]]] = {
    "frontier": [
        {"provider": "openrouter", "model": "anthropic/claude-4-sonnet-20250522", "label": "claude-4-sonnet"},
        {"provider": "openrouter", "model": "google/gemini-2.5-pro", "label": "gemini-2.5-pro"},
    ],
    "mid-range": [
        {"provider": "deepseek", "model": "deepseek-chat", "label": "deepseek-chat"},
        {"provider": "minimax", "model": "MiniMax-M2.7", "label": "minimax-m2.7"},
        {"provider": "openrouter", "model": "qwen/qwen3-coder-480b-a35b-07-25", "label": "qwen3-coder-480b"},
    ],
    "volume": [
        {"provider": "openrouter", "model": "openai/gpt-4.1-mini-2025-04-14", "label": "gpt-4.1-mini"},
        {"provider": "openrouter", "model": "google/gemini-2.5-flash", "label": "gemini-2.5-flash"},
    ],
}

# Detector → elicitor mapping (same as batch_elicit_multi.py)
DETECTOR_CONFIG = {
    "confabulation": {
        "import": "elicitation.elicit_confabulation",
        "function": "elicit_confabulation",
        "corpus_subdir": "confabulation",
        "default_steps": 6,
    },
    "thrash": {
        "import": "elicitation.elicit_thrash",
        "function": "elicit_thrash",
        "corpus_subdir": "thrash",
        "default_steps": 8,
    },
    "runaway_cost": {
        "import": "elicitation.elicit_runaway_cost",
        "function": "elicit_runaway_cost",
        "corpus_subdir": "runaway_cost",
        "default_steps": 8,
    },
}


def get_elicitor(detector: str):
    """Dynamically import and return the elicitor function."""
    cfg = DETECTOR_CONFIG[detector]
    mod = __import__(cfg["import"], fromlist=[cfg["function"]])
    return getattr(mod, cfg["function"])


def cross_validate_trace(snapshots, detector: str) -> dict:
    """Run cross-detector validation on a trace."""
    cfg = VitalsConfig(min_evidence_steps=3, model_size_class="auto")
    stuck_cross = False
    target_detected = False

    for i in range(2, len(snapshots)):
        current = snapshots[i]
        history = snapshots[:i]
        result = detect_loop(current, history, config=cfg, workflow_type="research")
        snapshot_dict = current.model_dump()
        snapshot_dict.update(result.as_snapshot_update())
        signals = derive_stop_signals(snapshot_dict, step_count=i + 1)

        if detector == "confabulation" and result.confabulation_detected:
            target_detected = True
        elif detector == "thrash" and signals.thrash_detected:
            target_detected = True
        elif detector == "runaway_cost" and signals.runaway_cost_detected:
            target_detected = True

        # v1.14.2: read explicit runaway_cost_detected instead of the
        # legacy stuck_trigger == "burn_rate_anomaly" sentinel string.
        if (
            detector != "stuck"
            and result.stuck_detected
            and not result.runaway_cost_detected
            and result.detector_priority != detector
        ):
            stuck_cross = True

    return {
        "target_detected": target_detected,
        "stuck_cross_fire": stuck_cross,
    }


def save_trace(run, detector: str, model_label: str, tier: str) -> Path:
    """Save trace to cross-model corpus directory."""
    subdir = "positive" if run.positive else "negative"
    trace_dir = (
        CORPUS_DIR
        / "traces"
        / DETECTOR_CONFIG[detector]["corpus_subdir"]
        / "elicited_cross_model"
        / subdir
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{run.trace_id}.json"
    data = [s.model_dump(mode="json") for s in run.snapshots]
    trace_path.write_text(json.dumps(data, indent=2))
    return trace_path


def build_manifest_entry(
    run, validation: dict, trace_path: Path, model_label: str, tier: str
) -> dict:
    """Build manifest entry with cross-model metadata."""
    confidence = run.metadata.confidence
    if validation["stuck_cross_fire"]:
        confidence = min(confidence, 0.5)

    return {
        "trace_id": run.trace_id,
        "path": str(trace_path.relative_to(CORPUS_DIR)),
        "tier": "elicited",
        "labels": run.metadata.labels,
        "metadata": {
            "generator": run.metadata.generator,
            "params": run.metadata.params,
            "onset_step": run.metadata.onset_step,
            "tier": "elicited",
            "model": model_label,
            "model_tier": tier,
            "provider": run.provider_name,
            "framework": None,
            "reviewer": None,
            "review_date": None,
            "confidence": confidence,
            "notes": f"cross-model campaign ({tier} tier)",
            "cross_validation": {
                "stuck_cross_fire": validation["stuck_cross_fire"],
                "target_detected": validation["target_detected"],
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model elicitation campaign (Sprint 9)"
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(MODEL_TIERS),
        choices=list(MODEL_TIERS),
        help="Model tiers to elicit from",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["confabulation", "thrash", "runaway_cost"],
        choices=list(DETECTOR_CONFIG),
    )
    parser.add_argument(
        "--traces-per-tier",
        type=int,
        default=20,
        help="Total positive traces per tier per detector (default: 20)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help="Ratio of negative to positive traces (default: 0.25)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    neg_per_tier = max(1, int(args.traces_per_tier * args.negative_ratio))

    if args.dry_run:
        print("Cross-model elicitation plan:")
        print(f"  Tiers: {args.tiers}")
        print(f"  Detectors: {args.detectors}")
        print(f"  Positive per tier per detector: {args.traces_per_tier}")
        print(f"  Negative per tier per detector: {neg_per_tier}")
        print()
        total = 0
        for tier in args.tiers:
            models = MODEL_TIERS[tier]
            per_model_pos = args.traces_per_tier // len(models)
            per_model_neg = max(1, neg_per_tier // len(models))
            per_model_total = per_model_pos + per_model_neg
            print(f"  {tier} ({len(models)} models, {per_model_pos}+/{per_model_neg}- each):")
            for m in models:
                print(f"    {m['label']} via {m['provider']}")
            total += per_model_total * len(models) * len(args.detectors)
        print(f"\n  Total traces: {total}")
        return

    # Load existing manifest
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    existing_ids = {e["trace_id"] for e in manifest}

    stats = {
        "total": 0,
        "passed": 0,
        "cross_fire": 0,
        "errors": 0,
        "skipped_providers": 0,
    }
    tier_stats: dict[str, dict[str, int]] = {}

    for tier in args.tiers:
        models = MODEL_TIERS[tier]
        per_model_pos = args.traces_per_tier // len(models)
        per_model_neg = max(1, neg_per_tier // len(models))

        for model_cfg in models:
            provider_name = model_cfg["provider"]
            model_id = model_cfg["model"]
            model_label = model_cfg["label"]

            print(f"\n{'='*60}")
            print(f"[{tier}] {model_label} via {provider_name}")
            print(f"{'='*60}")

            try:
                provider = get_provider(provider_name, model=model_id)
            except Exception as e:
                print(f"  SKIP provider ({provider_name}): {e}")
                stats["skipped_providers"] += 1
                continue

            for detector in args.detectors:
                elicit_fn = get_elicitor(detector)
                default_steps = DETECTOR_CONFIG[detector]["default_steps"]
                tier_key = f"{tier}/{model_label}/{detector}"
                tier_stats.setdefault(tier_key, {"pos": 0, "neg": 0})

                for polarity, count in [
                    ("positive", per_model_pos),
                    ("negative", per_model_neg),
                ]:
                    is_positive = polarity == "positive"
                    for i in range(count):
                        stats["total"] += 1
                        print(
                            f"  {detector} [{i+1}/{count}] {polarity}...",
                            end=" ",
                            flush=True,
                        )

                        try:
                            kwargs: dict = {
                                "positive": is_positive,
                                "total_steps": default_steps,
                            }
                            if detector == "confabulation":
                                kwargs["verify_api_key"] = os.environ.get(
                                    "SEMANTIC_SCHOLAR_API_KEY"
                                )
                            run = elicit_fn(provider, **kwargs)
                        except Exception as e:
                            stats["errors"] += 1
                            print(f"ERROR: {e}")
                            continue

                        if run.trace_id in existing_ids:
                            print("SKIP (dup)")
                            continue

                        validation = cross_validate_trace(run.snapshots, detector)
                        trace_path = save_trace(run, detector, model_label, tier)
                        entry = build_manifest_entry(
                            run, validation, trace_path, model_label, tier
                        )
                        manifest.append(entry)
                        existing_ids.add(run.trace_id)

                        conf = entry["metadata"]["confidence"]
                        pol_key = "pos" if is_positive else "neg"
                        tier_stats[tier_key][pol_key] += 1

                        if validation["stuck_cross_fire"]:
                            stats["cross_fire"] += 1
                            print(f"CROSS-FIRE conf={conf}")
                        else:
                            stats["passed"] += 1
                            print(f"OK conf={conf}")

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL ELICITATION SUMMARY")
    print(f"{'='*60}")
    print(
        f"Total: {stats['total']}  Passed: {stats['passed']}  "
        f"Cross-fire: {stats['cross_fire']}  Errors: {stats['errors']}  "
        f"Skipped providers: {stats['skipped_providers']}"
    )

    print("\nPer-tier breakdown:")
    for key in sorted(tier_stats):
        s = tier_stats[key]
        print(f"  {key}: {s['pos']}+ / {s['neg']}-")

    # Check success criteria
    print("\nSuccess criteria check:")
    tiers_represented = set()
    for key in tier_stats:
        tier = key.split("/")[0]
        if tier_stats[key]["pos"] > 0:
            tiers_represented.add(tier)
    print(f"  Tiers represented: {len(tiers_represented)} (target: 3+)")

    for detector in args.detectors:
        det_traces = sum(
            tier_stats[k]["pos"]
            for k in tier_stats
            if k.endswith(f"/{detector}")
        )
        status = "OK" if det_traces >= 20 else "BELOW TARGET"
        print(f"  {detector}: {det_traces} positive traces ({status})")


if __name__ == "__main__":
    main()
