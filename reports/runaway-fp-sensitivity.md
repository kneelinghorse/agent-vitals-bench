# Runaway_cost FP Sensitivity Analysis

**Corpus:** v1 | **Mode:** default | **Generated:** 2026-04-10 01:31 UTC

## Summary

- Total runaway_cost traces: 1494 (229 pos, 1265 neg)
- True Positives: 229 | False Positives: 52
- FPs analyzed at firing step: 52 | TPs analyzed: 229

---

## Question 1: Coverage at FP Firing Step

**0/52** FPs have coverage >= 0.50 at the firing step (0.0%)

Also: 37/52 have coverage >= 0.25 (71.2%)

| Coverage Bucket | Count | % of FPs |
|----------------|-------|----------|
| < 0.25 | 15 | 28.8% |
| 0.25-0.49 | 37 | 71.2% |
| 0.50-0.74 | 0 | 0.0% |
| >= 0.75 | 0 | 0.0% |

## Question 2: Findings Delta at FP Firing Step

**8/52** FPs have findings_delta >= 2 at the firing step (15.4%)

Also: 52/52 have findings_delta >= 1 (100.0%)

## Question 3: Burn Rate Ratio Distribution

Ratio = actual tokens-per-finding / baseline tokens-per-finding at firing step.
Detector fires when ratio > 2.5x baseline (default multiplier).

| Ratio Bucket | Count | % of FPs |
|-------------|-------|----------|
| 2.5x | 0 | 0.0% |
| 3.0x | 21 | 40.4% |
| 3.5x | 25 | 48.1% |
| 4.0x | 3 | 5.8% |
| 4.0+x | 3 | 5.8% |

## Question 4: Safety Check — TPs with coverage >= 0.50

**203/229** TPs have coverage >= 0.50 at firing step.

**UNSAFE** — Option A would suppress the following TPs:
  - runaway-syn-001 (cov=0.500)
  - runaway-syn-002 (cov=0.500)
  - runaway-syn-003 (cov=0.500)
  - runaway-syn-004 (cov=0.500)
  - runaway-syn-005 (cov=0.500)
  - runaway-syn-006 (cov=0.500)
  - runaway-syn-007 (cov=0.500)
  - runaway-syn-008 (cov=0.500)
  - runaway-syn-009 (cov=0.500)
  - runaway-syn-010 (cov=0.567)
  - runaway-syn-011 (cov=0.567)
  - runaway-syn-012 (cov=0.567)
  - runaway-syn-013 (cov=0.567)
  - runaway-syn-014 (cov=0.567)
  - runaway-syn-015 (cov=0.567)
  - runaway-syn-016 (cov=0.567)
  - runaway-syn-017 (cov=0.567)
  - runaway-syn-018 (cov=0.567)
  - runaway-syn-019 (cov=0.500)
  - runaway-syn-020 (cov=0.500)
  - runaway-syn-021 (cov=0.500)
  - runaway-syn-022 (cov=0.500)
  - runaway-syn-023 (cov=0.500)
  - runaway-syn-024 (cov=0.500)
  - runaway-syn-025 (cov=0.500)
  - runaway-syn-026 (cov=0.500)
  - runaway-syn-027 (cov=0.500)
  - runaway-syn-028 (cov=0.500)
  - runaway-syn-029 (cov=0.500)
  - runaway-syn-030 (cov=0.500)
  - runaway-syn-031 (cov=0.500)
  - runaway-syn-032 (cov=0.500)
  - runaway-syn-033 (cov=0.500)
  - runaway-syn-034 (cov=0.500)
  - runaway-syn-035 (cov=0.500)
  - runaway-syn-036 (cov=0.500)
  - runaway-syn-037 (cov=0.500)
  - runaway-syn-038 (cov=0.500)
  - runaway-syn-039 (cov=0.500)
  - runaway-syn-040 (cov=0.500)
  - runaway-syn-041 (cov=0.500)
  - runaway-syn-042 (cov=0.500)
  - runaway-syn-043 (cov=0.500)
  - runaway-syn-044 (cov=0.500)
  - runaway-syn-045 (cov=0.500)
  - runaway-syn-046 (cov=0.500)
  - runaway-syn-047 (cov=0.500)
  - runaway-syn-048 (cov=0.500)
  - runaway-syn-049 (cov=0.500)
  - runaway-syn-050 (cov=0.500)
  - runaway-syn-051 (cov=0.500)
  - runaway-syn-052 (cov=0.500)
  - runaway-syn-053 (cov=0.500)
  - runaway-syn-054 (cov=0.500)
  - runaway-syn-055 (cov=0.500)
  - runaway-syn-056 (cov=0.500)
  - runaway-syn-057 (cov=0.500)
  - runaway-syn-058 (cov=0.500)
  - runaway-syn-059 (cov=0.500)
  - runaway-syn-060 (cov=0.500)
  - runaway-syn-061 (cov=0.500)
  - runaway-syn-062 (cov=0.500)
  - runaway-syn-063 (cov=0.500)
  - runaway-syn-064 (cov=0.500)
  - runaway-syn-065 (cov=0.500)
  - runaway-syn-066 (cov=0.500)
  - runaway-syn-067 (cov=0.500)
  - runaway-syn-068 (cov=0.500)
  - runaway-syn-069 (cov=0.500)
  - runaway-syn-070 (cov=0.500)
  - runaway-syn-071 (cov=0.500)
  - runaway-syn-072 (cov=0.500)
  - runaway-elic-pos-872b1b87 (cov=0.500)
  - runaway-elic-pos-9be9f415 (cov=0.500)
  - runaway-elic-pos-a446c848 (cov=0.500)
  - runaway-elic-pos-510c4a05 (cov=0.500)
  - runaway-elic-pos-ec48d2b4 (cov=0.500)
  - runaway-elic-pos-deba52a3 (cov=0.500)
  - runaway-elic-pos-3a04608a (cov=0.500)
  - runaway-elic-pos-d2e88a82 (cov=0.500)
  - runaway-elic-pos-8b127a15 (cov=0.500)
  - runaway-elic-pos-fbae49df (cov=0.500)
  - runaway-elic-pos-cf617f82 (cov=0.500)
  - runaway-elic-pos-f8b43090 (cov=0.500)
  - runaway-elic-pos-7c88a2f5 (cov=0.500)
  - runaway-elic-pos-1106bdc8 (cov=0.500)
  - runaway-elic-pos-811c2ddd (cov=0.500)
  - runaway-elic-pos-58284801 (cov=0.500)
  - runaway-elic-pos-e74f8354 (cov=0.500)
  - runaway-elic-pos-ac874015 (cov=0.500)
  - runaway-elic-pos-33a9e3e0 (cov=0.500)
  - runaway-elic-pos-15a77247 (cov=0.500)
  - runaway-elic-pos-4e118df0 (cov=0.500)
  - runaway-elic-pos-bb0d8044 (cov=0.500)
  - runaway-elic-pos-5f6b975b (cov=0.500)
  - runaway-elic-pos-ee9842a3 (cov=0.500)
  - runaway-elic-pos-c4b68840 (cov=0.500)
  - runaway-elic-pos-9c0ec0af (cov=0.500)
  - runaway-elic-pos-fbcd33d2 (cov=0.500)
  - runaway-elic-pos-1692b602 (cov=0.500)
  - runaway-elic-pos-9a9a4b77 (cov=0.500)
  - runaway-elic-pos-9f1532b8 (cov=0.500)
  - runaway-elic-pos-e5c6822e (cov=0.500)
  - runaway-la-001 (cov=0.500)
  - runaway-la-002 (cov=0.500)
  - runaway-la-003 (cov=0.567)
  - runaway-la-004 (cov=0.567)
  - runaway-la-005 (cov=0.500)
  - runaway-la-006 (cov=0.500)
  - runaway-la-007 (cov=0.500)
  - runaway-la-008 (cov=0.500)
  - runaway-la-009 (cov=0.500)
  - runaway-la-010 (cov=0.500)
  - runaway-la-011 (cov=0.500)
  - runaway-la-012 (cov=0.500)
  - runaway-cr-001 (cov=0.500)
  - runaway-cr-002 (cov=0.500)
  - runaway-cr-003 (cov=0.567)
  - runaway-cr-004 (cov=0.567)
  - runaway-cr-005 (cov=0.500)
  - runaway-cr-006 (cov=0.500)
  - runaway-cr-007 (cov=0.500)
  - runaway-cr-008 (cov=0.500)
  - runaway-cr-009 (cov=0.500)
  - runaway-cr-010 (cov=0.500)
  - runaway-cr-011 (cov=0.500)
  - runaway-cr-012 (cov=0.500)
  - runaway-ds-001 (cov=0.500)
  - runaway-ds-002 (cov=0.500)
  - runaway-ds-003 (cov=0.567)
  - runaway-ds-004 (cov=0.567)
  - runaway-ds-005 (cov=0.500)
  - runaway-ds-006 (cov=0.500)
  - runaway-ds-007 (cov=0.500)
  - runaway-ds-008 (cov=0.500)
  - runaway-ds-009 (cov=0.500)
  - runaway-ds-010 (cov=0.500)
  - runaway-ds-011 (cov=0.500)
  - runaway-ds-012 (cov=0.500)
  - runaway-elic-pos-2b28e712 (cov=0.500)
  - runaway-elic-pos-15487256 (cov=0.500)
  - runaway-elic-pos-78e8a7f7 (cov=0.500)
  - runaway-elic-pos-e9493b44 (cov=0.500)
  - runaway-elic-pos-2232c6d5 (cov=0.500)
  - runaway-elic-pos-2c465103 (cov=0.500)
  - runaway-elic-pos-16611a0b (cov=0.500)
  - runaway-elic-pos-95f8f05e (cov=0.500)
  - runaway-elic-pos-daa7de9c (cov=0.500)
  - runaway-elic-pos-5680ace9 (cov=0.500)
  - runaway-elic-pos-f0ac515d (cov=0.500)
  - runaway-elic-pos-0f32c041 (cov=0.500)
  - runaway-elic-pos-b56facb8 (cov=0.500)
  - runaway-elic-pos-653bb594 (cov=0.500)
  - runaway-elic-pos-36165af5 (cov=0.500)
  - runaway-elic-pos-347c8fef (cov=0.500)
  - runaway-elic-pos-6c6aee19 (cov=0.500)
  - runaway-elic-pos-7776e48f (cov=0.500)
  - runaway-elic-pos-13cb61fc (cov=0.531)
  - runaway-elic-pos-2dfb3bbe (cov=0.500)
  - runaway-elic-pos-7c9e4c3d (cov=0.500)
  - runaway-elic-pos-81ae204d (cov=0.500)
  - runaway-elic-pos-f338b109 (cov=0.500)
  - runaway-elic-pos-65102ba5 (cov=0.500)
  - runaway-elic-pos-68dc9538 (cov=0.500)
  - runaway-elic-pos-f0d5967d (cov=0.500)
  - runaway-elic-pos-93205115 (cov=0.500)
  - runaway-elic-pos-2aa4c7df (cov=0.500)
  - runaway-elic-pos-2f3e8471 (cov=0.500)
  - runaway-elic-pos-551d35cd (cov=0.500)
  - runaway-elic-pos-10a866c9 (cov=0.500)
  - runaway-elic-pos-64588dc4 (cov=0.500)
  - runaway-elic-pos-c8c1e0ab (cov=0.500)
  - runaway-elic-pos-1dd26019 (cov=0.500)
  - runaway-elic-pos-7758865d (cov=0.500)
  - runaway-elic-pos-c73d0b94 (cov=0.500)
  - runaway-elic-pos-4fc8cd17 (cov=0.500)
  - runaway-elic-pos-b4263a65 (cov=0.500)
  - runaway-elic-pos-35f4c62a (cov=0.500)
  - runaway-elic-pos-11d5c22d (cov=0.500)
  - runaway-elic-pos-aa8f9d94 (cov=0.500)
  - runaway-elic-pos-d7120e07 (cov=0.500)
  - runaway-elic-pos-87569d07 (cov=0.500)
  - runaway-elic-pos-6b89d957 (cov=0.500)
  - runaway-elic-pos-4164b521 (cov=0.500)
  - runaway-elic-pos-e21c3893 (cov=0.500)
  - runaway-elic-pos-10ea5cf4 (cov=0.500)
  - runaway-elic-pos-c9fbb86b (cov=0.500)
  - runaway-elic-pos-4797bddd (cov=0.531)
  - runaway-elic-pos-15303c73 (cov=0.500)
  - runaway-elic-pos-df5553c6 (cov=0.500)
  - runaway-elic-pos-366fe7d5 (cov=0.500)
  - runaway-elic-pos-48caa137 (cov=0.500)
  - runaway-elic-pos-59d93a6b (cov=0.500)
  - runaway-elic-pos-e5cc220d (cov=0.500)
  - runaway-elic-pos-4df3ed7a (cov=0.531)
  - runaway-elic-pos-db22a453 (cov=0.500)
  - runaway-elic-pos-83e01307 (cov=0.500)
  - runaway-elic-pos-31a01ad2 (cov=0.531)
  - runaway-elic-pos-03f50b3e (cov=0.531)
  - runaway-elic-pos-dec7a355 (cov=0.531)
  - runaway-elic-pos-b3717a74 (cov=0.531)
  - runaway-elic-pos-f5a86b95 (cov=0.531)
  - runaway-elic-pos-124394da (cov=0.531)

---

## Proposed Rule Changes — Impact Assessment

| Option | Description | FPs Suppressed | FPs Remaining | TPs Lost | Safe? |
|--------|-------------|---------------|---------------|----------|-------|
| A | Coverage >= 0.50 suppression | 0/52 | 52 | 203/229 | NO |
| B | Multiplier 2.5 → 3.0 | 20/52 | 32 | 0/229 | YES |
| C | Min baseline depth >= 3 | 39/52 | 13 | 220/229 | REVIEW |
| A+B | Combined | 20/52 | 32 | 203/229 | REVIEW |

---

## FP Detail Table

| Trace ID | Step | Total | Coverage | Findings Δ | Token Δ | Ratio | Baseline Depth |
|----------|------|-------|----------|------------|---------|-------|----------------|
| thrash-elic-neg-711104d5 | 3 | 8 | 0.300 | 1.0 | 1155 | 2.52x | 2 |
| runaway-elic-neg-31dc9040 | 2 | 8 | 0.319 | 1.0 | 489 | 2.53x | 1 |
| runaway-elic-neg-8932d780 | 2 | 8 | 0.319 | 1.0 | 498 | 2.54x | 1 |
| confab-elic-neg-9f2c19f7 | 4 | 6 | 0.225 | 3.0 | 1008 | 2.55x | 3 |
| thrash-elic-neg-2b2ab20c | 6 | 8 | 0.450 | 1.0 | 2084 | 2.55x | 5 |
| thrash-elic-neg-ef13b991 | 6 | 8 | 0.450 | 1.0 | 1585 | 2.58x | 5 |
| runaway-elic-neg-5d6c2e7a | 2 | 8 | 0.319 | 1.0 | 486 | 2.59x | 1 |
| runaway-elic-neg-e45fe40b | 2 | 8 | 0.319 | 1.0 | 512 | 2.59x | 1 |
| runaway-elic-neg-3dffd05d | 2 | 8 | 0.319 | 1.0 | 502 | 2.61x | 1 |
| runaway-elic-neg-70b4d955 | 2 | 8 | 0.319 | 1.0 | 505 | 2.62x | 1 |
| confab-elic-pos-64de761d | 4 | 6 | 0.360 | 3.0 | 786 | 2.64x | 1 |
| thrash-elic-neg-09e45418 | 3 | 8 | 0.300 | 1.0 | 1261 | 2.67x | 2 |
| runaway-elic-neg-078c4bd2 | 2 | 8 | 0.319 | 1.0 | 531 | 2.71x | 1 |
| confab-elic-pos-9e5ecb6a | 2 | 6 | 0.280 | 8.0 | 1465 | 2.74x | 1 |
| confab-elic-pos-24458ed8 | 4 | 6 | 0.225 | 2.0 | 606 | 2.76x | 2 |
| runaway-elic-neg-f7e04f1a | 2 | 8 | 0.319 | 1.0 | 543 | 2.78x | 1 |
| runaway-elic-neg-4f7d06fa | 2 | 8 | 0.319 | 1.0 | 559 | 2.84x | 1 |
| runaway-elic-neg-ca5d0eb9 | 2 | 8 | 0.319 | 1.0 | 559 | 2.85x | 1 |
| runaway-elic-neg-2effbcf1 | 2 | 8 | 0.319 | 1.0 | 549 | 2.86x | 1 |
| confab-elic-pos-40302bc3 | 2 | 6 | 0.280 | 8.0 | 1589 | 2.92x | 1 |
| thrash-elic-neg-bd02b2dc | 6 | 8 | 0.450 | 1.0 | 1899 | 2.97x | 5 |
| thrash-elic-neg-2744bf7c | 3 | 8 | 0.300 | 1.0 | 1358 | 3.01x | 2 |
| stuck-syn-021 | 5 | 8 | 0.227 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-024 | 5 | 8 | 0.227 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-027 | 5 | 8 | 0.227 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-030 | 6 | 8 | 0.334 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-033 | 6 | 8 | 0.334 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-036 | 6 | 8 | 0.334 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-048 | 5 | 10 | 0.185 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-051 | 5 | 10 | 0.185 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-054 | 5 | 10 | 0.185 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-057 | 6 | 10 | 0.270 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-060 | 6 | 10 | 0.270 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-063 | 6 | 10 | 0.270 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-066 | 8 | 10 | 0.440 | 1.0 | 1750 | 3.50x | 4 |
| stuck-syn-069 | 8 | 10 | 0.440 | 1.0 | 1750 | 3.50x | 4 |
| stuck-syn-072 | 8 | 10 | 0.440 | 1.0 | 1750 | 3.50x | 4 |
| stuck-syn-075 | 5 | 12 | 0.157 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-078 | 5 | 12 | 0.157 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-081 | 5 | 12 | 0.157 | 1.0 | 1750 | 3.50x | 1 |
| stuck-syn-084 | 6 | 12 | 0.227 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-087 | 6 | 12 | 0.227 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-090 | 6 | 12 | 0.227 | 1.0 | 1750 | 3.50x | 2 |
| stuck-syn-093 | 8 | 12 | 0.369 | 1.0 | 1750 | 3.50x | 4 |
| stuck-syn-096 | 8 | 12 | 0.369 | 1.0 | 1750 | 3.50x | 4 |
| stuck-syn-099 | 8 | 12 | 0.369 | 1.0 | 1750 | 3.50x | 4 |
| confab-elic-pos-d28bd629 | 3 | 6 | 0.180 | 5.0 | 2145 | 3.61x | 1 |
| confab-elic-pos-f3c6ca8f | 3 | 6 | 0.320 | 1.0 | 2153 | 3.69x | 2 |
| confab-elic-pos-d0376b5f | 4 | 6 | 0.360 | 2.0 | 1531 | 3.91x | 1 |
| confab-elic-neg-cb17d42e | 5 | 6 | 0.270 | 2.0 | 2129 | 5.99x | 4 |
| confab-elic-neg-80220c47 | 5 | 6 | 0.270 | 1.0 | 934 | 6.10x | 4 |
| confab-elic-neg-1f413db1 | 5 | 6 | 0.270 | 1.0 | 1141 | 7.18x | 4 |
