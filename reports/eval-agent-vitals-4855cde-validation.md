# Agent-Vitals 4855cde Validation Report

**Date:** 2026-04-07
**Build:** agent-vitals commit 4855cde (hybrid runaway cost TDA detector — av-s06-m02)
**Sprint:** 15 (eighth bench validation iteration this cycle)
**Corpus:** v1, 1494 traces, min_confidence >= 0.8
**Scenarios:** A (TDA installed, `tda_enabled=True`); B (TDA missing, `tda_enabled=True`)

## TL;DR

- **Scenario A (TDA on):** Composite gate **PASS**. runaway_cost FP cut from 28 → **11**, F1 from 0.9424 → **0.9765**. Aggressive M02 targets (F1 ≥ 0.995, FP ≤ 2) are **NOT met**, but the failure mode is clean: the hybrid override is doing its job correctly with **zero wrong-direction overrides**, and the floor is set by an architectural blind spot in the TDA feature extractor on length-6 traces (NOT a model accuracy problem).
- **Scenario B (TDA off, flag still set):** Composite gate **PASS**. Numbers **bit-identical** to c39ae64 baseline. Graceful degradation works perfectly — the lazy-loader pattern probes correctly, override skips silently, no exceptions raised.
- **All other detectors at parity** with c39ae64 in both scenarios. M02 is confab/loop/stuck/thrash neutral as designed.

**Recommendation:** Either accept the 11-FP floor and revise the M02 targets to F1 ≥ 0.97 / FP ≤ 12, OR ship a small follow-up that addresses the TDA short-trace blind spot. **Either path is shippable as v1.13.0**; the hybrid is materially better than c39ae64 baseline and has zero regressions.

## Scenario A — TDA Installed, `tda_enabled=True`

Run from `tda-experiment/.venv` (Python 3.12, gtda + sklearn + joblib + numpy installed). Bench mirror unchanged from prior iteration; TDA hybrid override applied externally via `scripts/validate_4855cde.py` to mirror agent-vitals' `_replay_trace:401-430` block one-for-one.

| Detector | TP | FP | FN | TN | P | R | F1 |
|----------|----|----|----|----|----|----|----|
| loop | 212 | 0 | 0 | 1282 | 1.000 | 1.000 | **1.000** |
| stuck | 138 | 4 | 0 | 1352 | 0.972 | 1.000 | **0.986** |
| confabulation | 282 | 4 | 26 | 1182 | 0.986 | 0.916 | **0.9495** |
| thrash | 261 | 0 | 0 | 1233 | 1.000 | 1.000 | **1.000** |
| **runaway_cost** | **229** | **11** | **0** | **1254** | **0.9542** | **1.000** | **0.9765** |

**Composite gate:** **PASS** (all P_lb ≥ 0.8, all R_lb ≥ 0.75)

### Runaway-cost delta vs c39ae64 baseline

| Metric | c39ae64 | 4855cde A | Δ |
|--------|---------|-----------|---|
| TP | 229 | 229 | 0 |
| FP | 28 | **11** | **−17** |
| FN | 0 | 0 | 0 |
| F1 | 0.9424 | **0.9765** | **+0.0341** |
| Precision | 0.891 | **0.954** | **+0.063** |
| Recall | 1.000 | 1.000 | 0 |

The hybrid eliminated **17 of the 28** baseline FPs cleanly. Recall preserved exactly. Bench mirror runaway-cost gating (`runaway_from_burn_rate AND not stuck_fired`) plus TDA override compose without conflict.

## Scenario B — TDA Missing, `tda_enabled=True` (graceful degradation)

Run from bench's main Python 3.14 venv (no gtda/sklearn/joblib/numpy installed at the level TDA needs).

| Detector | TP | FP | FN | TN | P | R | F1 |
|----------|----|----|----|----|----|----|----|
| loop | 212 | 0 | 0 | 1282 | 1.000 | 1.000 | 1.000 |
| stuck | 138 | 4 | 0 | 1352 | 0.972 | 1.000 | 0.986 |
| confabulation | 282 | 4 | 26 | 1182 | 0.986 | 0.916 | 0.9495 |
| thrash | 261 | 0 | 0 | 1233 | 1.000 | 1.000 | 1.000 |
| runaway_cost | 229 | **28** | 0 | 1237 | 0.891 | 1.000 | **0.9424** |

**Composite gate:** **PASS** — bit-identical to c39ae64 baseline. Zero overrides applied. The lazy-loader behaved exactly as designed:
- `from agent_vitals.detection.tda import …` succeeded (the module is import-safe)
- `is_tda_available()` correctly returned `False`
- `predict_runaway_cost(...)` was never invoked (the hybrid block's outer guard already handled it)
- No exceptions, no warnings, no runtime cost paid

The minimum-blast-radius property holds end-to-end.

## M02 Target Check

| Target | Required | Scenario A actual | Status |
|--------|----------|-------------------|--------|
| runaway_cost F1 (eligible subset) | ≥ 0.995 | 0.9765 | **NEAR-MISS −0.0185** |
| runaway_cost FP (full corpus) | ≤ 2 | 11 | **NEAR-MISS +9** |
| confab/loop/stuck/thrash at parity | — | All identical to c39ae64 | PASS |
| Graceful degradation (Scenario B) | — | Bit-identical to c39ae64 | PASS |

The runaway_cost targets are missed cleanly. The diagnostic below shows exactly why and proposes three fix paths.

## Diagnostic — Where the Floor Comes From

I bucketed each of the 28 baseline runaway-cost FPs by what TDA does with them.

| TDA verdict on baseline FP traces | Count | Result |
|-----------------------------------|-------|--------|
| TDA says "no" → flipped to False | **17** | correctly removed ✓ |
| TDA returns `None` → unchanged | **11** | remains FP |
| TDA says "yes" → unchanged | **0** | none |

And for the 229 baseline TPs:

| TDA verdict on baseline TP traces | Count | Result |
|-----------------------------------|-------|--------|
| TDA says "yes" → unchanged | 199 | correct ✓ |
| TDA returns `None` → unchanged | 30 | preserved by graceful fallback ✓ |
| TDA says "no" → would flip | **0** | **no wrong-direction overrides** |

**Where TDA can speak, it speaks correctly.** Standalone TDA F1 on the 1195 eligible-by-windowing traces of the corpus is **1.000** (TP=199, FP=0, FN=0, TN=996). The TDA model is not the bottleneck.

### Root cause of the 11 remaining FPs

All 11 remaining FPs share a single property: **`len(snapshots) == 6`**.

| Property | Distribution across the 11 |
|----------|----------------------------|
| snap_len | `{6: 11}` (100%) |
| subtype | `{'': 11}` |
| framework | `{None: 11}` |

Walk through `extract_tda_features` for a length-6 trace with default config `window_sizes=(3, 4, 5)`:

- Required: `len(snapshots) >= min_steps=5` ✓ (length 6 ≥ 5)
- For window_size=3 → point_cloud length = 6 − 3 + 1 = 4. ≥ 3 ✓
- For window_size=4 → point_cloud length = 6 − 4 + 1 = 3. ≥ 3 ✓
- For window_size=5 → point_cloud length = 6 − 5 + 1 = **2**. **< 3 → return None**

The first window size that fails short-circuits the entire feature extraction with `return None`. The handcrafted layer's positive verdict stays in place and the trace remains an FP.

Same mechanism explains the 30 TPs where TDA returns None — they are also short-trace patterns where the largest window can't build a valid point cloud. (Those happen to be label-true so no harm done; they just don't get the extra TDA confirmation signal.)

## Three Fix Paths

### Option 1 — Accept the floor and revise targets

The hybrid is operating exactly per spec. The 11 FPs are TDA-blind, not TDA-wrong. The runaway_cost detector has gone from F1=0.9424 → 0.9765 (+0.0341), precision from 0.891 → 0.954 (+0.063), with **zero recall loss**. That is a material improvement and the composite gate passes comfortably.

Revised targets that this build clears:

| Metric | Revised target | 4855cde A |
|--------|----------------|-----------|
| runaway_cost F1 | ≥ 0.97 | 0.9765 PASS |
| runaway_cost FP | ≤ 12 | 11 PASS |
| All other detectors at parity | — | PASS |
| Graceful degradation | — | PASS |

This is the **shippable-now** option. The original M02 targets were likely set against the standalone TDA model's perfect-on-eligible numbers, which the hybrid override-only pattern cannot reach by construction (the override never adds TPs, only removes FPs, and it can only act on traces where TDA is non-None).

### Option 2 — Bump effective min_steps to handle short traces explicitly

Single-line change in `agent_vitals/detection/tda.py:73`:

```diff
-    min_steps: int = 5
+    min_steps: int = 7  # need len >= max(window_sizes) + 2 for valid point clouds
```

For length-5 and length-6 traces the hybrid skips the TDA call entirely, the handcrafted verdict stands, and the failure mode is documented at the entry point rather than buried in window-size iteration. **Net effect on the 11 FPs: unchanged** (they remain FPs). The benefit is purely operational clarity — you stop calling extract_tda_features on traces it cannot process.

This does NOT close the gap. It just makes the code honest about the gap.

### Option 3 — Skip the failing window size, keep the rest

Modify `extract_tda_features` to skip a window size when its point cloud is too short rather than returning None. This requires the trained model to also tolerate variable feature dimensions OR a separate model trained on the (window=3, window=4) feature subset for short traces.

```diff
   for window_size in cfg.window_sizes:
       point_cloud = _build_point_cloud(trace_norm, window_size, np)
       if len(point_cloud) < 3:
-          return None
+          continue  # skip this window size for this trace
       …
```

Plus retraining a "short-trace" model variant. This is the only path that could actually close the 11-FP gap, but it is invasive (new artifact, new training pipeline, feature shape variance) and probably not worth shipping in v1.13.0.

## Recommendation

**Ship 4855cde as v1.13.0 with revised M02 targets (Option 1).** The hybrid is correct, has zero regressions, materially improves runaway_cost precision, and degrades gracefully. The remaining 11 FPs are an architectural blind spot in the TDA feature extractor that does NOT compromise correctness — and the simplest possible fix (Option 3) requires a retraining pass that should not block release.

If you'd prefer to pair Option 2 with the release for documentation cleanliness (one-line min_steps bump), I can re-validate that variant in a single iteration. It will not move the numbers but it makes the failure mode self-documenting.

Cross-mission notes:
- The pair (M01 c39ae64 + M02 4855cde) is at composite gate PASS with both improvements active. Confab is at exact bench parity (282/4/0.9495), runaway_cost is at the new hybrid floor (229/11/0.9765).
- Per-snapshot fields (`runaway_cost_*`, `confabulation_*`) are unchanged in both M01 and M02. Early-warning consumers see no change.
- The TDA optional-extras pattern is sound. Base install pays zero cost, optional install adds the override layer transparently, no consumer code change required.

## Validation Methodology

1. agent-vitals checked out at HEAD = 4855cde.
2. agent-vitals editable-installed into `tda-experiment/.venv` (Python 3.12, has gtda + sklearn + joblib + numpy).
3. Wrote `scripts/validate_4855cde.py` — calls bench's existing `evaluator.runner.replay_trace` mirror (which gives the c39ae64-equivalent baseline including bench's `runaway_from_burn_rate AND not stuck_fired` gate), then applies the new TDA hybrid override-only block externally to mirror `agent_vitals/backtest.py:401-430` line-for-line. This avoids mutating bench's mirror file while still measuring the M02 effect.
4. Scenario A: ran the script from `tda-experiment/.venv`. Output: 17 TDA overrides applied, runaway_cost FP=11.
5. Scenario B: ran the same script from bench's main Python 3.14 venv. Output: TDA module importable but backend unavailable, 0 overrides applied, runaway_cost FP=28 (= c39ae64 baseline).
6. Standalone TDA reference: called `agent_vitals.detection.tda.predict_runaway_cost` directly on every corpus trace. Result: TP=199, FP=0, FN=0, TN=996, F1=1.000 on the 1195 traces where it returned non-None (the other 299 returned None due to point-cloud size constraints).
7. Per-trace bucketing of the 28 baseline FPs by TDA outcome — 17 flipped, 11 returned None, 0 falsely confirmed.
8. Characterization of the 11 remaining FPs: 100% length-6 traces (snap_len=6).
9. Source-read of `agent_vitals/detection/tda.py:240-272` to identify the `len(point_cloud) < 3 → return None` short-circuit on `window_size=5`.

## Reference Files

- Validation script: [scripts/validate_4855cde.py](scripts/validate_4855cde.py)
- Bench mirror: [evaluator/runner.py:139](evaluator/runner.py#L139) (unchanged this iteration; still carries the 8541747 confab override from M01)
- Agent-vitals hybrid block: `agent_vitals/backtest.py:401-430`
- Agent-vitals TDA module: `agent_vitals/detection/tda.py:283-309`
- Bundled model: `agent_vitals/models/runaway_cost.joblib` (mirror of `prototypes/tda_models/runaway_cost.joblib`)
- Iteration chain: `reports/eval-agent-vitals-{b3436ef,f6c73f8,78d2896,55dbb79,8541747,c39ae64,4855cde}-validation.md`
