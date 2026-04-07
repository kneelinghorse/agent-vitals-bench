# Agent-Vitals b3436ef Validation Report

**Date:** 2026-04-06
**Build:** agent-vitals commit b3436ef (causal confabulation detector, Paths 1-2 only)
**Sprint:** 15 (bench validation of upstream implementation)
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## Summary

The agent-vitals causal confabulation detector implementation **passes the composite HARD GATE** but **does not meet the implementation request targets**. The gap is explained by the missing **Path 3 (verified_source_decoupling)** added in bench Sprint 15 (s15-m01). Independent of Path 3, the implementation also produces more false positives than the bench reference prototype, suggesting an integration or threshold issue.

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|------|------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.982 | 0.982 | 212 | HARD GATE |
| stuck | 0.979 | 1.000 | 0.989 | 0.939 | 0.973 | 138 | HARD GATE |
| confabulation | 0.855 | 0.864 | 0.860 | 0.812 | 0.821 | 308 | HARD GATE |
| thrash | 1.000 | 1.000 | 1.000 | 0.986 | 0.986 | 261 | HARD GATE |
| runaway_cost | 0.891 | 1.000 | 0.942 | 0.847 | 0.984 | 229 | HARD GATE |

**Composite gate:** PASS

## Confabulation Detail

**Confusion matrix (b3436ef):** TP=266, FP=45, FN=42, TN=1141

### Against Implementation Request Targets

| Metric | Target | b3436ef | Status |
|--------|--------|---------|--------|
| Confab F1 (full) | >= 0.990 | 0.860 | FAIL (-0.130) |
| Confab precision (full) | >= 0.98 | 0.855 | FAIL (-0.125) |
| Confab recall (full) | >= 0.96 | 0.864 | FAIL (-0.096) |
| Full corpus FP count | <= 2 | 45 | FAIL (+43) |

### Against Bench Reference Prototype (with Path 3)

| Metric | Bench prototype | b3436ef | Delta |
|--------|----------------|---------|-------|
| TP | 282 | 266 | -16 |
| FP | 4 | 45 | +41 |
| FN | 26 | 42 | +16 |
| F1 | 0.950 | 0.860 | -0.090 |
| P_lb | 0.965 | 0.812 | -0.153 |
| R_lb | 0.879 | 0.821 | -0.058 |

## Root Cause Analysis

The 90-point F1 gap decomposes into two distinct issues:

### Issue 1: Missing Path 3 (accounts for ~16 of the 16 FN delta)

The b3436ef implementation only contains Paths 1-2 from spec v1. Path 3 (verified_source_decoupling), added in bench Sprint 15 after the original spec was delivered, catches real LLM confabulation in elicited traces. Without it, elicited positives where findings_count and sources_count grow in lockstep are missed.

**Fix:** Implement Path 3 from spec v2. The bench reference prototype recovered 62 TP with zero new FP using this path on the full corpus. See updated `specs/causal-confab-detector.md`.

### Issue 2: Excess False Positives (+41 FP vs bench prototype)

The b3436ef implementation produces 45 FP on the full corpus, while the bench reference prototype produces only 4 FP under the same configuration. Possible causes:

1. **Guard interaction:** The added "causal path skipped when sources_count is always 0" guard may be interacting with the existing SFR fallback in unexpected ways. If the SFR fallback also fires on the same trace, you get double-counting.
2. **Threshold drift:** Verify that the implemented config defaults exactly match spec values: window_size=4, baseline_floor=0.4, weak_link_threshold=0.35, structural_drop_threshold=0.2, ratio_gate=0.4, low_link_threshold=0.2, source_bootstrap_cap=10, low_link_ratio_gate=0.3.
3. **Residualization implementation:** Verify the OLS residualization handles the zero-variance control case (return mean-centered target) and the unequal-length case (return target unchanged).
4. **Pearson correlation:** Verify None-handling when either sequence has zero variance.

**Recommended action:** Diff the implemented `_detect_confabulation_candidates()` causal path against `prototypes/causal_confab.py` in agent-vitals-bench. The reference prototype is 600 lines, self-contained, and produces P_lb=0.965 on the same corpus.

## No-Regression Verification (Other Detectors)

| Detector | Pre-b3436ef baseline | b3436ef | Delta | Status |
|----------|---------------------|---------|-------|--------|
| loop | F1=1.000 | F1=1.000 | 0 | OK |
| stuck | F1=0.991 (FP=2) | F1=0.989 (FP=3) | -0.002 | Minor |
| thrash | F1=1.000 | F1=1.000 | 0 | OK |
| runaway_cost | F1=0.959 (FP=17) | F1=0.942 (FP=28) | -0.017 | Slight FP regression |

The runaway_cost FP regression (+11 FP) is unexpected — the causal confab change should not affect runaway_cost detection. Worth investigating whether the changes to `detection/loop.py` accidentally touch shared utilities.

## Recommendations for Next b3436ef+1 Build

1. **Implement Path 3** from updated spec v2 (`specs/causal-confab-detector.md`). This is the largest single improvement available and requires only the existing `verified_sources_count` / `unverified_sources_count` schema fields.

2. **Diff causal path against bench reference** to find the source of the +41 FP gap. The bench prototype `prototypes/causal_confab.py` is a self-contained reference that achieves P_lb=0.965 on the same corpus.

3. **Investigate the runaway_cost FP regression** — the +11 FP delta is unexpected and suggests the change touched shared code.

4. **Re-run bench validation** after fixes. Targets:
   - Confab F1 >= 0.945 on full corpus (matches bench prototype)
   - Confab FP count <= 4 on full corpus
   - Runaway_cost FP count back to 17 or below
   - All other detectors at parity with pre-b3436ef baseline

## Reference Files

- Updated spec: `specs/causal-confab-detector.md` (v2 with Path 3)
- Bench reference prototype: `prototypes/causal_confab.py`
- Per-model elicitation validation: `reports/eval-per-model-causal-validation.md`
- Bench v1 corpus: `corpus/v1/`
