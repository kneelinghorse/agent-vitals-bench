"""Ad-hoc validation harness for agent-vitals 4855cde (M02 TDA hybrid).

Uses bench's canonical production-replay wrapper in explicit ``default``
mode to get the non-TDA baseline, then applies the TDA hybrid override-only
block externally (mirroring the diff in ``agent_vitals.backtest``) so the
release-branch validation remains reproducible.

Run from either:
  - tda-experiment venv (TDA available)  → Scenario A
  - bench main venv (TDA missing)        → Scenario B
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent_vitals import VitalsConfig  # noqa: E402
from evaluator.runner import (  # noqa: E402
    DETECTORS,
    load_manifest,
    load_trace,
    replay_trace,
    resolve_workflow_type,
)


def _try_tda():
    try:
        from agent_vitals.detection.tda import (
            MissingTDADependencyError,
            TDAConfig,
            is_tda_available,
            predict_runaway_cost,
        )
    except ImportError:
        return None
    return {
        "available": is_tda_available(),
        "TDAConfig": TDAConfig,
        "MissingTDADependencyError": MissingTDADependencyError,
        "predict_runaway_cost": predict_runaway_cost,
    }


def main() -> int:
    cfg = VitalsConfig.from_yaml()
    tda = _try_tda()
    tda_available = bool(tda and tda["available"])
    print(f"tda module importable: {tda is not None}")
    print(f"tda backend available: {tda_available}")

    manifest = load_manifest("v1")
    counts = {d: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for d in DETECTORS}
    n_traces = 0
    n_runaway_overrides = 0  # how many traces did TDA flip false?
    n_runaway_eligible_short = 0  # how many runaway-fired traces below min_steps?

    for entry in manifest:
        meta_conf = entry.get("metadata", {}).get("confidence", 0.0)
        if meta_conf < 0.8:
            continue
        snaps = load_trace("v1", entry["path"])
        wf = resolve_workflow_type(entry["trace_id"], entry.get("mission_id"))
        preds = replay_trace(
            snaps,
            config=cfg,
            workflow_type=wf,
            runtime_mode="default",
        )
        n_traces += 1

        # Apply TDA hybrid override-only (mirrors agent-vitals/backtest.py:410-430)
        if preds["runaway_cost"] and tda is not None and tda_available:
            if len(snaps) >= 7:
                try:
                    tda_cfg = tda["TDAConfig"]()
                    tda_pred = tda["predict_runaway_cost"](snaps, config=tda_cfg)
                except (tda["MissingTDADependencyError"], FileNotFoundError):
                    tda_pred = None
                if tda_pred is not None and not tda_pred.detected:
                    preds["runaway_cost"] = False
                    n_runaway_overrides += 1
            else:
                n_runaway_eligible_short += 1

        for d in DETECTORS:
            pred = bool(preds[d])
            label = bool(entry["labels"].get(d, False))
            key = "tp" if pred and label else "fp" if pred else "fn" if label else "tn"
            counts[d][key] += 1

    print()
    print(f"traces evaluated: {n_traces}")
    print(f"TDA overrides applied: {n_runaway_overrides}")
    print(f"runaway-fired traces below TDA min_steps=7: {n_runaway_eligible_short}")
    print()
    print(
        f"{'detector':14s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>5s} "
        f"{'P':>7s} {'R':>7s} {'F1':>7s}"
    )
    for d in DETECTORS:
        c = counts[d]
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(
            f"{d:14s} {c['tp']:>4d} {c['fp']:>4d} {c['fn']:>4d} {c['tn']:>5d} "
            f"{p:>7.4f} {r:>7.4f} {f1:>7.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
