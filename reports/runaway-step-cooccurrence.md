# Step-Level Detector Co-occurrence Analysis

**Corpus:** v1 | **Mode:** default | **agent-vitals:** v1.16.0 | **Generated:** 2026-04-10 03:33 UTC

## Summary

- **Total remaining FPs:** 30
- FPs where stuck also fires: 24
  - Same-step co-occurrence (stuck+runaway at same step): **0**
  - Temporally separated (stuck and runaway at different steps): **24**
- FPs without stuck co-fire: 6
- Mixed-label traces (stuck=true AND runaway=true): 0

---

## Co-occurrence Pattern Summary

| Pattern | Count | Suppression-Safe? |
|---------|-------|-------------------|
| Same-step co-occurrence | 0 | YES — window=0 suffices |
| Temporal separation | 24 | Depends on gap size |
| No stuck co-fire | 6 | N/A — different root cause |

---

## FP Trace Details

| Trace ID | Steps | Stuck fires at | Runaway fires at | Overlap | Gap |
|----------|-------|---------------|-----------------|---------|-----|
| stuck-syn-021 | 8 | 3,4,6,7 | 5 | — | 1 |
| stuck-syn-024 | 8 | 3,4,6,7 | 5 | — | 1 |
| stuck-syn-027 | 8 | 3,4,6,7 | 5 | — | 1 |
| stuck-syn-030 | 8 | 4,5,7 | 6 | — | 1 |
| stuck-syn-033 | 8 | 4,5,7 | 6 | — | 1 |
| stuck-syn-036 | 8 | 4,5,7 | 6 | — | 1 |
| stuck-syn-048 | 10 | 3,4,6,7,8,9 | 5 | — | 1 |
| stuck-syn-051 | 10 | 3,4,6,7,8,9 | 5 | — | 1 |
| stuck-syn-054 | 10 | 3,4,6,7,8,9 | 5 | — | 1 |
| stuck-syn-057 | 10 | 3,4,5,7,8 | 6,9 | — | 1 |
| stuck-syn-060 | 10 | 3,4,5,7,8 | 6,9 | — | 1 |
| stuck-syn-063 | 10 | 4,5,7,8 | 6,9 | — | 1 |
| stuck-syn-066 | 10 | 5,6,7,9 | 8 | — | 1 |
| stuck-syn-069 | 10 | 5,6,7,9 | 8 | — | 1 |
| stuck-syn-072 | 10 | 5,6,7,9 | 8 | — | 1 |
| stuck-syn-075 | 12 | 3,4,6,7,8,9,10,11 | 5 | — | 1 |
| stuck-syn-078 | 12 | 3,4,6,7,8,9,10,11 | 5 | — | 1 |
| stuck-syn-081 | 12 | 3,4,6,7,8,9,10,11 | 5 | — | 1 |
| stuck-syn-084 | 12 | 3,4,5,7,8,10,11 | 6,9 | — | 1 |
| stuck-syn-087 | 12 | 3,4,5,7,8,10,11 | 6,9 | — | 1 |
| stuck-syn-090 | 12 | 4,5,7,8,10,11 | 6,9 | — | 1 |
| stuck-syn-093 | 12 | 5,6,7,9,10 | 8,11 | — | 1 |
| stuck-syn-096 | 12 | 5,6,7,9,10 | 8,11 | — | 1 |
| stuck-syn-099 | 12 | 5,6,7,9,10 | 8,11 | — | 1 |
| thrash-elic-neg-2744bf7c | 8 | — | 3 | — | — |
| confab-elic-neg-1f413db1 | 6 | — | 5 | — | — |
| confab-elic-neg-80220c47 | 6 | — | 5 | — | — |
| confab-elic-pos-d28bd629 | 6 | — | 3 | — | — |
| confab-elic-pos-f3c6ca8f | 6 | — | 3 | — | — |
| confab-elic-pos-d0376b5f | 6 | — | 4 | — | — |
