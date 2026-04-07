# Agent-Vitals 8541747 Validation Report

**Date:** 2026-04-06
**Build:** agent-vitals commit 8541747 (final-step adjudication for Path 3 — av-s06-m01)
**Sprint:** 15 (fifth bench validation iteration)
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## TL;DR — FAIL

The 8541747 fix is implemented exactly as designed and does eliminate the 11 transient FPs as predicted, but it ALSO suppresses 71 valid TPs on short synthetic confab traces. The result regresses confab F1 from 0.944 → **0.808** and fails the composite gate.

**Root cause:** `agent_vitals/detection/loop.py:_detect_causal_confabulation` carries hardcoded `latest_link <= threshold` / `latest_verified <= threshold` gates on Paths 1, 2, and 3 (added in 78d2896 as a per-step transient suppressor). With the new trace-level override doing the streaming-vs-one-shot reconciliation at the tail, those latest-window gates are now **double-counted suppression** that the prototype reference does not apply. The fix in 8541747 is correct in spirit; the underlying detector needs the latest-window gates removed to restore parity with the bench prototype reference.

**Recommendation:** Do not ship 8541747 as-is. Pair it with removal of the per-path latest-window gates (one-line deletes in three places in `detection/loop.py`). With that pairing, the bench prototype path predicts F1 ≈ 0.9495 (= bench reference one-shot), which matches the 8541747 commit's stated target.

## Gate Results (8541747)

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|------|------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.982 | 0.982 | 212 | HARD GATE |
| stuck | 0.972 | 1.000 | 0.986 | 0.930 | 0.973 | 138 | HARD GATE |
| confabulation | 0.986 | 0.685 | **0.808** | 0.960 | **0.631** | 308 | **NO-GO** |
| thrash | 1.000 | 1.000 | 1.000 | 0.986 | 0.986 | 261 | HARD GATE |
| runaway_cost | 0.891 | 1.000 | 0.942 | 0.847 | 0.984 | 229 | HARD GATE |

**Composite gate:** **FAIL** (confab `recall_lb = 0.631 < 0.75`)

All other detectors at parity with 55dbb79 — the change is confab-only as designed.

## Confab Detail: 55dbb79 → 8541747

| Metric | 55dbb79 | 8541747 (predicted by AV) | 8541747 (actual) | Δ vs prediction |
|--------|---------|---------------------------|------------------|-----------------|
| TP | 288 | ~282 | **211** | **−71** |
| FP | 14 | ~4 | **3** | −1 |
| FN | 20 | ~26 | **97** | **+71** |
| F1 | 0.944 | ~0.950 | **0.808** | **−0.142** |
| Precision | 0.954 | ~0.986 | 0.986 | 0 |
| Recall | 0.935 | ~0.916 | **0.685** | **−0.231** |

The override correctly removed 11 of the 14 FPs. It also removed 71 valid TPs. Precision matches the prediction exactly; recall collapsed.

## Diagnostic — Why So Many TPs Were Lost

I implemented the override in two places to cross-check:
1. **Bench evaluator mirror** ([evaluator/runner.py:139](evaluator/runner.py#L139)) — added `final_confabulation_detected` capture and the `if predictions["confabulation"] and not final_confabulation_detected: predictions["confabulation"] = False` override, mirroring the diff in 8541747 line-for-line.
2. **Direct call** to `agent_vitals.backtest._replay_trace` from a script, on the same corpus.

Both paths produce **identical** numbers (TP=211, FP=3, FN=97, TN=1183). My mirror is bit-correct and the divergence is real, not a port bug.

I then ran a per-trace comparison between agent-vitals' `_replay_trace` (with override) and bench's prototype reference `prototypes/causal_confab.py:detect_causal_confabulation`:

| | Count | Notes |
|---|---|---|
| Both fire | 214 | |
| Both don't fire | 1208 | |
| **prototype TRUE / agent-vitals FALSE** | **72** | **agent-vitals strict subset of prototype** |
| prototype FALSE / agent-vitals TRUE | 0 | none |

Of the 72 traces where agent-vitals suppressed and prototype fired:
- **71 are real confabs** (label=True, lost TPs)
- **1 is a label-false** (legitimate FP that override correctly removes)
- All 72 are synthetic `confab-syn-*` traces with snapshot length **6, 8, 10, or 12** (short)
- snap-len distribution: `{8: 30, 10: 25, 12: 16, 6: 1}`

Zero traces flip the other direction. This is a strict containment: agent-vitals' detector predictions are a strict subset of prototype predictions on this corpus.

## Root Cause — Latest-Window Gates Hardcoded in Always-On Path

Side-by-side comparison of Path 1 (causal_link_break):

**Prototype** ([prototypes/causal_confab.py:391-396](prototypes/causal_confab.py#L391-L396)):

```python
structural_break = (
    baseline_window.link_strength >= effective_config.baseline_floor
    and weakest_window.link_strength <= effective_config.weak_link_threshold
    and structural_drop >= effective_config.structural_drop_threshold
    and final_ratio <= effective_config.ratio_gate
)
```

**Agent-vitals** (`agent_vitals/detection/loop.py:591-597`):

```python
structural_break = (
    baseline_strength >= float(cfg.causal_confab_baseline_floor)
    and weakest_strength <= float(cfg.causal_confab_weak_link_threshold)
    and latest_link <= float(cfg.causal_confab_weak_link_threshold)   # ← EXTRA
    and structural_drop >= float(cfg.causal_confab_structural_drop_threshold)
    and final_ratio <= float(cfg.causal_confab_ratio_gate)
)
```

Same shape on Path 2 (`latest_link <= low_link_threshold` extra) and Path 3 (`latest_verified <= verified_link_floor` extra at line 636).

These three latest-window gates were introduced in 78d2896 as a per-step transient suppressor — and they did exactly that job per their original commit intent. The problem now: they sit in the **always-on** path of `_detect_causal_confabulation`, so they apply at the final tail call as well as at every per-step call.

8541747's `_replay_trace` override assumes the final tail call's verdict is equivalent to the bench reference one-shot. It is not, because the latest-window gates impose a stricter condition than the prototype: "the trace must currently be in a weak state at the very last window," vs the prototype's "the trace's strongest window is below floor (whole-trace)."

For the 71 short synthetic confab traces, the trace IS in a confab state by the prototype's criterion (whole-trace baseline below floor), but the very last window happens to score above the latest-window gate due to noise — short traces with `window_size=4` have only 3-9 windows, so any single-window noise is enough to flip the gate.

## What Actually Closes The Gap

**Fix:** Remove the three `latest_link` / `latest_verified` checks from `_detect_causal_confabulation` (one line each on Paths 1/2/3). With 8541747's `_replay_trace` override now reconciling streaming-vs-one-shot at the trace level, those gates are redundant suppression.

```diff
   structural_break = (
       baseline_strength >= float(cfg.causal_confab_baseline_floor)
       and weakest_strength <= float(cfg.causal_confab_weak_link_threshold)
-      and latest_link <= float(cfg.causal_confab_weak_link_threshold)
       and structural_drop >= float(cfg.causal_confab_structural_drop_threshold)
       and final_ratio <= float(cfg.causal_confab_ratio_gate)
   )
   persistent_low = (
       weakest_strength <= float(cfg.causal_confab_low_link_threshold)
-      and latest_link <= float(cfg.causal_confab_low_link_threshold)
       and initial_sources <= int(cfg.causal_confab_source_bootstrap_cap)
       and final_ratio <= float(cfg.causal_confab_low_link_ratio_gate)
   )
   ...
   verified_decoupling = (
       verified_baseline_strength <= float(cfg.causal_confab_verified_link_floor)
       and verified_ratio <= float(cfg.causal_confab_verified_ratio_gate)
       and total_sources >= int(cfg.causal_confab_verified_min_sources)
-      and latest_verified <= float(cfg.causal_confab_verified_link_floor)
   )
```

**Expected impact** (verified by direct prototype run on this corpus):

| Metric | 8541747 | 8541747 + gate removal | Bench reference (proto) |
|--------|---------|------------------------|-------------------------|
| TP | 211 | 282 | 282 |
| FP | 3 | 4 | 4 |
| F1 | 0.808 | 0.9495 | 0.9495 |
| Precision | 0.986 | 0.986 | 0.986 |
| Recall | 0.685 | 0.9156 | 0.9156 |

This pairs cleanly with 8541747 — the override is the right architectural fix; the leftover latest-window gates are the bug.

### Why this is safe

The original concern that motivated 78d2896's latest-window gates was that streaming per-step replay aggregated transient firings via `confabulation_fired = any step`. 8541747 fixes that at the trace label layer with the override. Per-snapshot `confabulation_detected` flags are still produced as before — early-warning consumers see them streaming. The latest-window gates were a per-step suppressor for the same problem, now solved better at a different layer. Removing them does not re-introduce the original transient-FP problem because the trace label is no longer driven by any-step aggregation — it is driven by the final-step verdict, which the override authoritatively controls.

### Alternative if you want to keep the gates

If you'd rather keep the latest-window gates as a per-step early-warning conservatism, you could pass a flag through `_DetectionContext` (e.g., `is_final_adjudication=True`) and skip the gates only on the final tail call. This is more invasive than the three-line delete, and the override already provides the same safety, so the delete is the cleaner option.

## Validation Methodology

1. Updated bench's `evaluator/runner.py` `replay_trace` mirror with the same final-step override diff as 8541747 (uncommitted; `git diff` available on request).
2. Ran `python -m evaluator.runner --corpus v1 --detectors all --min-confidence 0.8` against the v1 corpus (1494 traces, min_confidence ≥ 0.8).
3. Cross-checked by calling `agent_vitals.backtest._replay_trace` directly on the same corpus from a one-shot script — bit-identical results to bench mirror.
4. Ran prototype reference `python -m prototypes.causal_confab --corpus v1 --min-confidence 0.8` to confirm the bench reference numbers (TP=282, FP=4, F1=0.9495).
5. Per-trace diff between agent-vitals predictions and prototype predictions to characterize divergence (72 strict-subset suppressions, all on short synthetic confabs).
6. Read `agent_vitals/detection/loop.py:_detect_causal_confabulation` to identify the three latest-window gates.

## Reference Files

- Bench mirror with override: `evaluator/runner.py:139` (uncommitted)
- Bench prototype reference: `prototypes/causal_confab.py:detect_causal_confabulation`
- Agent-vitals integrated detector: `agent_vitals/detection/loop.py:_detect_causal_confabulation`
- Agent-vitals replay loop: `agent_vitals/backtest.py:_replay_trace`
- Previous validations: `reports/eval-agent-vitals-{b3436ef,f6c73f8,78d2896,55dbb79}-validation.md`
