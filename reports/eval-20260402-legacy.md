# Agent Vitals Bench — Gate Report

**Corpus:** legacy
**Traces evaluated:** 299
**Detectors:** loop, stuck
**Generated:** 2026-04-02 01:26 UTC

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|----- |------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.984 | 0.984 | 233 | **HARD GATE** |
| stuck | 1.000 | 0.899 | 0.947 | 0.962 | 0.828 | 109 | **HARD GATE** |

**Composite gate:** PASS

## Confusion Matrices

### loop
- TP=233, FP=0, FN=0, TN=66
- Precision CI: [0.984, 1.000]
- Recall CI: [0.984, 1.000]

### stuck
- TP=98, FP=0, FN=11, TN=190
- Precision CI: [0.962, 1.000]
- Recall CI: [0.828, 0.943]

## Gate Check Details

### loop
- min_positives: 233 (required: 25) — pass
- precision_lb: 0.9838 (required: 0.8) — pass
- recall_lb: 0.9838 (required: 0.75) — pass

### stuck
- min_positives: 109 (required: 25) — pass
- precision_lb: 0.9623 (required: 0.8) — pass
- recall_lb: 0.8283 (required: 0.75) — pass
