# Agent Vitals Bench — Gate Report

**Corpus:** v1
**Profile:** crewai
**Runtime Mode:** default
**Traces evaluated:** 1334
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost
**Generated:** 2026-04-10 22:20 UTC

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|----- |------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.980 | 0.980 | 184 | **HARD GATE** |
| stuck | 1.000 | 1.000 | 1.000 | 0.969 | 0.969 | 121 | **HARD GATE** |
| confabulation | 0.984 | 0.904 | 0.943 | 0.960 | 0.864 | 272 | **HARD GATE** |
| thrash | 1.000 | 1.000 | 1.000 | 0.984 | 0.984 | 237 | **HARD GATE** |
| runaway_cost | 0.972 | 1.000 | 0.986 | 0.939 | 0.982 | 205 | **HARD GATE** |

**Composite gate:** PASS

## Confusion Matrices

### loop
- TP=184, FP=0, FN=0, TN=1150
- Precision CI: [0.980, 1.000]
- Recall CI: [0.980, 1.000]

### stuck
- TP=121, FP=0, FN=0, TN=1213
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
- TP=205, FP=6, FN=0, TN=1123
- Precision CI: [0.939, 0.987]
- Recall CI: [0.982, 1.000]

## Gate Check Details

### loop
- min_positives: 184 (required: 25) — pass
- precision_lb: 0.9795 (required: 0.8) — pass
- recall_lb: 0.9795 (required: 0.75) — pass

### stuck
- min_positives: 121 (required: 25) — pass
- precision_lb: 0.9692 (required: 0.8) — pass
- recall_lb: 0.9692 (required: 0.75) — pass

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
- precision_lb: 0.9394 (required: 0.8) — pass
- recall_lb: 0.9816 (required: 0.75) — pass
