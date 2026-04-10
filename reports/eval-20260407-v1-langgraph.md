# Agent Vitals Bench — Gate Report

**Corpus:** v1
**Profile:** langgraph
**Traces evaluated:** 1334
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost
**Generated:** 2026-04-07 14:50 UTC

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|----- |------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.980 | 0.980 | 184 | **HARD GATE** |
| stuck | 1.000 | 1.000 | 1.000 | 0.969 | 0.969 | 120 | **HARD GATE** |
| confabulation | 0.984 | 0.904 | 0.943 | 0.960 | 0.864 | 272 | **HARD GATE** |
| thrash | 1.000 | 1.000 | 1.000 | 0.984 | 0.984 | 237 | **HARD GATE** |
| runaway_cost | 0.880 | 1.000 | 0.936 | 0.832 | 0.982 | 205 | **HARD GATE** |

**Composite gate:** PASS

## Confusion Matrices

### loop
- TP=184, FP=0, FN=0, TN=1150
- Precision CI: [0.980, 1.000]
- Recall CI: [0.980, 1.000]

### stuck
- TP=120, FP=0, FN=0, TN=1214
- Precision CI: [0.969, 1.000]
- Recall CI: [0.969, 1.000]

### confabulation
- TP=246, FP=4, FN=26, TN=1058
- Precision CI: [0.960, 0.994]
- Recall CI: [0.864, 0.934]

### thrash
- TP=237, FP=0, FN=0, TN=1097
- Precision CI: [0.984, 1.000]
- Recall CI: [0.984, 1.000]

### runaway_cost
- TP=205, FP=28, FN=0, TN=1101
- Precision CI: [0.832, 0.916]
- Recall CI: [0.982, 1.000]

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
- min_positives: 272 (required: 25) — pass
- precision_lb: 0.9596 (required: 0.8) — pass
- recall_lb: 0.8636 (required: 0.75) — pass

### thrash
- min_positives: 237 (required: 25) — pass
- precision_lb: 0.984 (required: 0.8) — pass
- recall_lb: 0.984 (required: 0.75) — pass

### runaway_cost
- min_positives: 205 (required: 25) — pass
- precision_lb: 0.8318 (required: 0.8) — pass
- recall_lb: 0.9816 (required: 0.75) — pass
