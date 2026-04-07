# Agent Vitals Bench — Gate Report

**Corpus:** v1
**Profile:** crewai
**Traces evaluated:** 953
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost
**Generated:** 2026-04-03 21:42 UTC

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|----- |------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.980 | 0.980 | 184 | **HARD GATE** |
| stuck | 1.000 | 1.000 | 1.000 | 0.969 | 0.969 | 120 | **HARD GATE** |
| confabulation | 0.981 | 0.883 | 0.929 | 0.944 | 0.826 | 171 | **HARD GATE** |
| thrash | 1.000 | 1.000 | 1.000 | 0.973 | 0.973 | 139 | **HARD GATE** |
| runaway_cost | 0.978 | 1.000 | 0.989 | 0.938 | 0.972 | 134 | **HARD GATE** |

**Composite gate:** PASS

## Confusion Matrices

### loop
- TP=184, FP=0, FN=0, TN=769
- Precision CI: [0.980, 1.000]
- Recall CI: [0.980, 1.000]

### stuck
- TP=120, FP=0, FN=0, TN=833
- Precision CI: [0.969, 1.000]
- Recall CI: [0.969, 1.000]

### confabulation
- TP=151, FP=3, FN=20, TN=779
- Precision CI: [0.944, 0.993]
- Recall CI: [0.826, 0.923]

### thrash
- TP=139, FP=0, FN=0, TN=814
- Precision CI: [0.973, 1.000]
- Recall CI: [0.973, 1.000]

### runaway_cost
- TP=134, FP=3, FN=0, TN=816
- Precision CI: [0.938, 0.993]
- Recall CI: [0.972, 1.000]

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
- min_positives: 171 (required: 25) — pass
- precision_lb: 0.9443 (required: 0.8) — pass
- recall_lb: 0.8263 (required: 0.75) — pass

### thrash
- min_positives: 139 (required: 25) — pass
- precision_lb: 0.9731 (required: 0.8) — pass
- recall_lb: 0.9731 (required: 0.75) — pass

### runaway_cost
- min_positives: 134 (required: 25) — pass
- precision_lb: 0.9376 (required: 0.8) — pass
- recall_lb: 0.9721 (required: 0.75) — pass
