# Causal vs TDA Comparative Benchmark

**Generated:** 2026-04-06 18:34 UTC
**Corpus:** v1
**Min confidence:** 0.8
**Filtered traces:** 1494
**TDA-eligible traces:** 1195
**TDA exclusions:** 299

## Coverage and Cross-Validation Setup

- Handcrafted and causal predictions are scored locally on the bench corpus.
- TDA uses 5-fold stratified cross-validation through the sibling Python 3.12 `tda-experiment` environment.
- Comparative metrics, agreement matrices, and failure analysis use the TDA-eligible subset so every approach is compared on the same traces.
- TDA exclusions by reason: short_trace_dropout=299

## Per-Detector Comparison

| Detector | Approach | Coverage | F1 | P_lb | R_lb | TP | FP | FN |
|----------|----------|----------|----|------|------|----|----|----|
| loop | handcrafted | 1494/1494 | 1.000 | 0.982 | 0.982 | 212 | 0 | 0 |
| loop | tda | 1195/1494 | 0.993 | 0.960 | 0.982 | 212 | 3 | 0 |
| stuck | handcrafted | 1494/1494 | 0.991 | 0.936 | 0.966 | 108 | 2 | 0 |
| stuck | tda | 1195/1494 | 0.986 | 0.936 | 0.949 | 107 | 2 | 1 |
| confabulation | handcrafted | 1494/1494 | 0.979 | 0.925 | 0.982 | 213 | 9 | 0 |
| confabulation | causal | 1494/1494 | 0.995 | 0.982 | 0.966 | 211 | 0 | 2 |
| confabulation | tda | 1195/1494 | 0.991 | 0.974 | 0.959 | 210 | 1 | 3 |
| thrash | handcrafted | 1494/1494 | 1.000 | 0.984 | 0.984 | 231 | 0 | 0 |
| thrash | tda | 1195/1494 | 1.000 | 0.984 | 0.984 | 231 | 0 | 0 |
| runaway_cost | handcrafted | 1494/1494 | 0.959 | 0.878 | 0.981 | 199 | 17 | 0 |
| runaway_cost | causal | 1494/1494 | 0.936 | 0.832 | 0.981 | 199 | 27 | 0 |
| runaway_cost | tda | 1195/1494 | 1.000 | 0.981 | 0.981 | 199 | 0 | 0 |

## Recommendations

| Detector | Recommendation | Why | Hybrid Option |
|----------|----------------|-----|---------------|
| loop | handcrafted | Handcrafted remains effectively tied with the best alternative on the aligned subset while avoiding training and optional TDA dependencies. | Keep TDA as an offline audit path rather than the primary detector. |
| stuck | handcrafted | Handcrafted remains effectively tied with the best alternative on the aligned subset while avoiding training and optional TDA dependencies. | Keep TDA as an offline audit path rather than the primary detector. |
| confabulation | causal | Causal posts the strongest aligned metrics on this detector and is the best candidate for upstreaming if the added complexity is acceptable. | None |
| thrash | handcrafted | Handcrafted remains effectively tied with the best alternative on the aligned subset while avoiding training and optional TDA dependencies. | Keep TDA as an offline audit path rather than the primary detector. |
| runaway_cost | tda | The selected approach improves the runaway-cost lower bounds on the aligned subset, which is the key weakness of the handcrafted threshold path. | Retain handcrafted burn-rate checks as a cheap first-pass screen and use the tda path as the adjudication layer. |

## Agreement and Divergence

### loop
- Aligned traces: 1195
- Pairwise agreement handcrafted vs tda: 0.997 (1192/1195)
- Positive-trace signatures: handcrafted+tda=212
- Negative-trace signatures: none=980, tda=3
- Divergence example `runaway-syn-061` (traces/runaway_cost/positive/runaway-syn-061.json, label=False): handcrafted=False, tda=True
- Divergence example `runaway-syn-062` (traces/runaway_cost/positive/runaway-syn-062.json, label=False): handcrafted=False, tda=True
- Divergence example `runaway-syn-063` (traces/runaway_cost/positive/runaway-syn-063.json, label=False): handcrafted=False, tda=True

### stuck
- Aligned traces: 1195
- Pairwise agreement handcrafted vs tda: 0.999 (1194/1195)
- Positive-trace signatures: handcrafted+tda=107, handcrafted=1
- Negative-trace signatures: none=1085, handcrafted+tda=2
- Divergence example `stuck-syn-059` (traces/stuck/positive/stuck-syn-059.json, label=True): handcrafted=True, tda=False

### confabulation
- Aligned traces: 1195
- Pairwise agreement handcrafted vs causal: 0.991 (1184/1195)
- Pairwise agreement handcrafted vs tda: 0.989 (1182/1195)
- Pairwise agreement causal vs tda: 0.995 (1189/1195)
- Positive-trace signatures: handcrafted+causal+tda=208, handcrafted+causal=3, handcrafted+tda=2
- Negative-trace signatures: none=972, handcrafted=9, tda=1
- Divergence example `confab-syn-094` (traces/confabulation/positive/confab-syn-094.json, label=True): handcrafted=True, causal=True, tda=False
- Divergence example `confab-syn-boundary-122` (traces/confabulation/positive/confab-syn-boundary-122.json, label=True): handcrafted=True, causal=False, tda=True
- Divergence example `confab-syn-boundary-123` (traces/confabulation/positive/confab-syn-boundary-123.json, label=True): handcrafted=True, causal=False, tda=True
- Divergence example `thrash-elic-neg-12c81079` (traces/thrash/elicited_cross_model/negative/thrash-elic-neg-12c81079.json, label=False): handcrafted=False, causal=False, tda=True
- Divergence example `confab-syn-delayed-002` (traces/confabulation/positive/confab-syn-delayed-002.json, label=True): handcrafted=True, causal=True, tda=False
- Divergence example `confab-syn-delayed-025` (traces/confabulation/positive/confab-syn-delayed-025.json, label=True): handcrafted=True, causal=True, tda=False

### thrash
- Aligned traces: 1195
- Pairwise agreement handcrafted vs tda: 1.000 (1195/1195)
- Positive-trace signatures: handcrafted+tda=231
- Negative-trace signatures: none=964

### runaway_cost
- Aligned traces: 1195
- Pairwise agreement handcrafted vs causal: 0.963 (1151/1195)
- Pairwise agreement handcrafted vs tda: 0.986 (1178/1195)
- Pairwise agreement causal vs tda: 0.977 (1168/1195)
- Positive-trace signatures: handcrafted+causal+tda=199
- Negative-trace signatures: none=952, causal=27, handcrafted=17
- Divergence example `stuck-syn-021` (traces/stuck/positive/stuck-syn-021.json, label=False): handcrafted=False, causal=True, tda=False
- Divergence example `stuck-syn-024` (traces/stuck/positive/stuck-syn-024.json, label=False): handcrafted=False, causal=True, tda=False
- Divergence example `stuck-syn-027` (traces/stuck/positive/stuck-syn-027.json, label=False): handcrafted=False, causal=True, tda=False
- Divergence example `stuck-syn-030` (traces/stuck/positive/stuck-syn-030.json, label=False): handcrafted=False, causal=True, tda=False
- Divergence example `stuck-syn-033` (traces/stuck/positive/stuck-syn-033.json, label=False): handcrafted=False, causal=True, tda=False
- Divergence example `stuck-syn-036` (traces/stuck/positive/stuck-syn-036.json, label=False): handcrafted=False, causal=True, tda=False

## Failure Mode Analysis

### loop
- handcrafted: FN=0, FP=0
- tda: FN=0, FP=3
- tda FP buckets: synthetic=3

### stuck
- handcrafted: FN=0, FP=2
- handcrafted FP buckets: framework=2
- tda: FN=1, FP=2
- tda FN buckets: synthetic=1
- tda FP buckets: framework=2

### confabulation
- handcrafted: FN=0, FP=9
- handcrafted FP buckets: delayed_onset=9
- causal: FN=2, FP=0
- causal FN buckets: baseline_never_established=2
- tda: FN=3, FP=1
- tda FN buckets: delayed_onset=2, synthetic=1
- tda FP buckets: elicited_cross_model=1

### thrash
- handcrafted: FN=0, FP=0
- tda: FN=0, FP=0

### runaway_cost
- handcrafted: FN=0, FP=17
- handcrafted FP buckets: elicited=15, elicited_cross_model=2
- causal: FN=0, FP=27
- causal FP buckets: cost_output_decoupling=27
- tda: FN=0, FP=0

