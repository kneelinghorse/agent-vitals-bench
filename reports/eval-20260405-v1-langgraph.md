# Agent Vitals Bench — Gate Report

**Corpus:** v1
**Profile:** langgraph
**Traces evaluated:** 984
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost
**Generated:** 2026-04-05 02:28 UTC

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|----- |------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.980 | 0.980 | 184 | **HARD GATE** |
| stuck | 1.000 | 1.000 | 1.000 | 0.969 | 0.969 | 120 | **HARD GATE** |
| confabulation | 0.974 | 0.884 | 0.927 | 0.936 | 0.827 | 172 | **HARD GATE** |
| thrash | 1.000 | 1.000 | 1.000 | 0.975 | 0.975 | 149 | **HARD GATE** |
| runaway_cost | 0.869 | 1.000 | 0.930 | 0.808 | 0.973 | 139 | **HARD GATE** |

**Composite gate:** PASS

## Confusion Matrices

### loop
- TP=184, FP=0, FN=0, TN=800
- Precision CI: [0.980, 1.000]
- Recall CI: [0.980, 1.000]

### stuck
- TP=120, FP=0, FN=0, TN=864
- Precision CI: [0.969, 1.000]
- Recall CI: [0.969, 1.000]

### confabulation
- TP=152, FP=4, FN=20, TN=808
- Precision CI: [0.936, 0.990]
- Recall CI: [0.827, 0.923]

### thrash
- TP=149, FP=0, FN=0, TN=835
- Precision CI: [0.975, 1.000]
- Recall CI: [0.975, 1.000]

### runaway_cost
- TP=139, FP=21, FN=0, TN=824
- Precision CI: [0.808, 0.913]
- Recall CI: [0.973, 1.000]

## Gate Check Details

### loop
- min_positives: 184 (required: 25) — pass
- precision_lb: 0.9795 (required: 0.8) — pass
- recall_lb: 0.9795 (required: 0.75) — pass

### stuck
- min_positives: 120 (required: 25) — pass
- precision_lb: 0.969 (required: 0.8) — pass
- recall_lb: 0.969 (required: 0.75) — pass

### confabulation
- min_positives: 172 (required: 25) — pass
- precision_lb: 0.9359 (required: 0.8) — pass
- recall_lb: 0.8272 (required: 0.75) — pass

### thrash
- min_positives: 149 (required: 25) — pass
- precision_lb: 0.9749 (required: 0.8) — pass
- recall_lb: 0.9749 (required: 0.75) — pass

### runaway_cost
- min_positives: 139 (required: 25) — pass
- precision_lb: 0.8077 (required: 0.8) — pass
- recall_lb: 0.9731 (required: 0.75) — pass
