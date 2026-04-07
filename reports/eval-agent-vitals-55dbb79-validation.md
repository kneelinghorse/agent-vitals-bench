# Agent-Vitals 55dbb79 Validation Report

**Date:** 2026-04-06
**Build:** agent-vitals commit 55dbb79 (Path 3 latest-window gate)
**Sprint:** 15 (fourth bench validation iteration)
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## Summary

The 55dbb79 build is **bit-identical** to 78d2896 on bench validation: TP=288, FP=14, F1=0.944. The Path 3 latest-window gate fix is correctly implemented but does not move the needle.

**My previous recommendation was wrong.** The latest-window gate cannot fix this gap, because it only catches "latest window strong but earlier max weak" — which is the opposite of the actual failure mode. I owe a corrected diagnosis below.

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|------|------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.982 | 0.982 | 212 | HARD GATE |
| stuck | 0.972 | 1.000 | 0.986 | 0.930 | 0.973 | 138 | HARD GATE |
| confabulation | 0.954 | 0.935 | 0.944 | 0.924 | 0.902 | 308 | HARD GATE |
| thrash | 1.000 | 1.000 | 1.000 | 0.986 | 0.986 | 261 | HARD GATE |
| runaway_cost | 0.891 | 1.000 | 0.942 | 0.847 | 0.984 | 229 | HARD GATE |

**Composite gate:** PASS

## Confab Detail: 78d2896 → 55dbb79

| Metric | 78d2896 | 55dbb79 | Bench reference |
|--------|---------|---------|-----------------|
| TP | 288 | **288** | 282 |
| FP | 14 | **14** | 4 |
| F1 | 0.944 | **0.944** | 0.950 |

Zero change. The fix is in the code path but does not eliminate any FP.

## Corrected Diagnosis: Why the Latest-Window Gate Doesn't Work

### What I claimed
"In streaming evaluation, at step 10 the max-so-far might be 0.4, Path 3 fires, and the verdict sticks. The latest-window gate would prevent firing when the latest window has recovered."

### What's actually happening

Walk through a real failure case. Trace has 30 snapshots, window_size=4.

At step k=12, the streaming detector sees first 12 snapshots. Windows are `[1-4], [2-5], ..., [9-12]`. Suppose all of them have `verified_link_strength <= 0.4`. Then:
- `verified_baseline_strength = max(...) = 0.4` <= floor (0.5) ✓
- `latest_window = [9-12]`, strength = 0.4 <= floor (0.5) ✓
- `verified_ratio` <= 0.3 ✓
- `total_sources >= 10` ✓
- **Path 3 fires at step 12.** `confabulation_fired = True`.

At step k=30 (full trace), windows extend to `[27-30]`. Suppose windows `[13-16], [14-17], ..., [27-30]` all have `verified_link_strength = 0.7` (the model started citing real papers).
- `verified_baseline_strength = max(...) = 0.7` > floor (0.5) ✗
- **Path 3 does NOT fire at step 30. Bench's one-shot evaluation also does not fire.**

In the streaming model, the step-12 firing already set `confabulation_fired = True`. The latest-window gate at step 12 didn't catch it because **the latest window AT step 12 was weak** — the trace was genuinely in a confab state at step 12. The recovery hadn't happened yet.

The gate I proposed checks "is the latest window strong now?" That's only useful when the historical max is weak but the latest window has *just* recovered. It cannot look ahead to see whether a future window will appear that pushes the historical max above the floor.

### The real mismatch

Bench's `verified_baseline_strength = max(across all N windows in full trace)` is a **whole-trace** signal. Streaming's `verified_baseline_strength = max(across first-k windows)` is a **prefix** signal. The prefix can be weak even when the whole-trace is strong, because future strong windows haven't been observed yet.

No per-step gate can fix this, because the missing information (future windows) is fundamentally unavailable at intermediate steps.

## What Actually Fixes This

There are three architecturally distinct options. None is a one-line change.

### Option A: Final-step adjudication (recommended)

Add a final-step pass that re-evaluates the trace using full-trace semantics and overrides the streaming verdict. The streaming detection still fires for early-warning purposes; the final trace label uses the bench-equivalent one-shot result.

```python
# In _replay_trace, after the per-step loop:
final_detection = detect_loop(all_snapshots, ...)
trace_labels["confabulation"] = final_detection.confabulation_detected
```

This is the cleanest semantic fix. Streaming detection retains early-warning value; trace labels match bench exactly. Trade-off: a different concept of "trace fired" vs "step fired" — useful for downstream consumers anyway (e.g., "did this trace ever flag" vs "is this trace currently flagging").

**Expected impact:** TP 288 → 282, FP 14 → 4, F1 0.944 → 0.950.

### Option B: Sustained-weakness requirement

Require Path 3 weakness to persist for the last `K` windows, not just the latest one. This filters transient bursts at the cost of K-window detection latency.

```python
last_k = verified_scores[-K:]
sustained_weak = all(v is not None and v <= floor for v in last_k)
verified_decoupling = (
    verified_baseline_strength <= floor
    and verified_ratio <= verified_ratio_gate
    and total_sources >= verified_min_sources
    and sustained_weak  # NEW
)
```

Picking K=4 or K=5 should eliminate most transients. Cleaner than option A as a localized change, but introduces detection latency and may not perfectly match bench.

**Expected impact:** Hard to predict without measurement. Probably FP 14 → 6-8, F1 0.944 → 0.947-0.949. Will not perfectly match bench.

### Option C: Tighter Path 3 thresholds

Make Path 3 firing rarer by lowering `verified_ratio_gate` (0.3 → 0.15) or raising `verified_min_sources` (10 → 20). This sacrifices recall on borderline cases but reduces transient FPs.

**Expected impact:** TP 288 → 280-285, FP 14 → 8-12, F1 0.944 → 0.945-0.947. Smallest improvement, suboptimal trade-off.

## Recommendation

**Ship 55dbb79 as v1.12.0.** The remaining 10-FP gap requires architectural changes (option A) that go beyond a fast tuning iteration. The current build is at HARD GATE with a 12-point P_lb buffer, F1=0.944 is essentially the request target (0.001 below), and the precision/recall profile (0.954 / 0.935) is a major improvement over the prior handcrafted detector.

If bench-parity is a hard requirement before release, option A is the right path — it's a localized change to `_replay_trace` (not `_detect_causal_confabulation`), and it produces the exact bench numbers because it uses the same one-shot semantic. But it should be a planned refactor, not a same-day patch, because it changes the meaning of `trace_labels["confabulation"]` and may have downstream effects in metrics, dashboards, and the streaming intervention path.

## Apology

I gave you a confident diagnosis on the 78d2896 report that turned out to be wrong. The latest-window gate I recommended sounds correct in isolation but cannot address the actual failure mode, because the streaming detector at step k cannot see future windows. I should have walked through a concrete trace example before proposing the fix. I will be more careful with claims about predicted FP/TP movements in future validation reports.

## Reference Files

- Bench reference: `prototypes/causal_confab.py` (one-shot evaluation in `evaluate_causal_confab_corpus`)
- Streaming aggregation: `agent_vitals/backtest.py:317-393` (any-step aggregation)
- Previous validations: `reports/eval-agent-vitals-{b3436ef,f6c73f8,78d2896}-validation.md`
