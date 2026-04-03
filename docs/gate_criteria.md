# Gate Criteria and Statistical Foundations

**Version:** 0.1
**Source:** AV-34 gate methodology (`agent-vitals/agent_vitals/ci_gate.py`)

---

## 1. Gate Promotion Criteria

A detector is **HARD GATE** (promoted) when ALL of the following hold:

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| Wilson CI precision lower bound | P_lb ≥ 0.80 | 95% confidence that true precision ≥ 80% |
| Wilson CI recall lower bound | R_lb ≥ 0.75 | 95% confidence that true recall ≥ 75% |
| Minimum positives | ≥ 25 accepted | Prevents promotion on tiny samples |

The composite gate passes when:
- F1 = 2·P·R / (P+R) ≥ 0.90 across all gate-passing detectors
- At least loop and stuck are at HARD GATE (minimum viable detection suite)

A detector that fails these criteria is **NO-GO** — it may not be cited as validated in Paper C.

---

## 2. Wilson Score Confidence Interval

The Wilson interval provides coverage-correct confidence bounds for a Bernoulli proportion. It is preferred over the naive (Wald) interval because it performs well at small sample sizes and near boundary values (p near 0 or 1).

**Formula:**
```
p̂ = successes / trials
z = 1.96 (95% confidence)

center = (p̂ + z²/2n) / (1 + z²/n)
spread = z × sqrt((p̂(1-p̂) + z²/4n) / n) / (1 + z²/n)

lower = center - spread
upper = center + spread
```

**For precision:** `successes = TP`, `trials = TP + FP`
**For recall:** `successes = TP`, `trials = TP + FN`

The implementation is in `agent-vitals/agent_vitals/ci_gate.py`:
```python
wilson_interval(successes=tp, trials=tp+fp)  # precision CI
wilson_interval(successes=tp, trials=tp+fn)  # recall CI
```

**Do not re-implement this in the bench.** Import and call it directly.

---

## 3. Minimum Sample Size Table

How many positives are needed to hit P_lb ≥ 0.80 at various observed precision levels?

| Observed precision | Positives for P_lb ≥ 0.80 | Positives for P_lb ≥ 0.75 |
|-------------------|--------------------------|--------------------------|
| 1.00 (perfect) | 16 | 12 |
| 0.95 | 25 | 18 |
| 0.90 | 40 | 28 |
| 0.85 | 70 | 45 |
| 0.80 | ∞ (barely meets) | 100+ |

**Practical target:** Collect 40+ positives per detector. This gives comfortable margin even with a few false positives in the elicited corpus.

---

## 4. AV-34 Baseline (Reference)

These are the results this bench must reproduce for loop and stuck, and must exceed for confabulation, thrash, runaway_cost.

Corpus: 421 evaluated / 340 reviewed. Workflow: build.mas DeepSearch + build. Models: 12 providers.

| Detector | TP | FP | FN | TN | P | R | P_lb | R_lb | Status |
|----------|----|----|----|----|---|---|------|------|--------|
| loop | — | — | — | — | 0.988 | 1.000 | — | — | HARD GATE |
| stuck | — | — | — | — | 0.919 | 0.838 | — | — | HARD GATE |
| confabulation | — | — | — | — | 0.957 | 1.000 | 0.790 | — | NO-GO |
| thrash | — | — | — | — | 1.000 | 1.000 | 0.741 | — | NO-GO (0 positives) |
| runaway_cost | — | — | — | — | 1.000 | 0.600 | 0.758 | 0.387 | NO-GO |
| composite | — | — | — | — | 1.000 | 0.965 | — | — | PASS |

*Note: TP/FP/FN/TN counts were not preserved in the summary report. Use these precision/recall values as calibration targets.*

**Confabulation NO-GO interpretation:** High precision (0.957) but P_lb = 0.790, just below the 0.80 threshold. The detector logic is correct — it just needs ~15 more validated positives to cross the Wilson CI threshold.

**Thrash NO-GO interpretation:** The detector fired on synthetic-quality traces during development, but 0 real-world positives were accepted in the AV-34 corpus review. Build.mas traces did not produce reliable thrash behavior. The elicitation harness exists specifically to fill this gap.

**Runaway_cost NO-GO interpretation:** Precision is fine (1.000) but recall is low (0.600) with wide CI (R_lb = 0.387). The detector misses genuine runaway cost events — either the trigger threshold is too conservative or the corpus positives were edge cases. Need to investigate whether this is a detector gap or a corpus gap before elicitation.

---

## 5. Gate Evaluation Run

The standard evaluation command:

```bash
make evaluate CORPUS=v1
```

Which runs:
```python
from evaluator.runner import run_evaluation
from evaluator.gate import check_all_gates

results = run_evaluation(corpus_version="v1", min_confidence=0.8)
gates = check_all_gates(results, min_positives=25, min_precision_lb=0.80, min_recall_lb=0.75)
```

Output: `reports/eval-{YYYYMMDD}-v1.md` with:
- Per-detector confusion matrix
- Wilson CI bounds at 95%
- Gate pass/fail per detector
- Composite gate result
- Provider breakdown (if model metadata present)

---

## 6. Cross-Model Reporting

Following the AV-34 model tier taxonomy, gate results should be reported per tier:

| Tier | Models |
|------|--------|
| Frontier | claude-sonnet-4-6, gpt-4o |
| Mid-range | Qwen3.5-72B (local), llama-3-70b, claude-haiku-4-5 |
| Volume | Qwen3.5-7B (local), gpt-4o-mini, gemini-flash |

The per-tier report answers: "Does the detector work equally well on cheap local models vs. frontier models?" If not, it surfaces a calibration gap.

---

## 7. Regression Gate

Every time `agent-vitals` is updated, run:

```bash
make evaluate CORPUS=legacy DETECTORS=loop,stuck
```

This confirms the update did not regress the two hard-gate detectors. A regression is any result where:
- P drops by > 3% from AV-34 baseline, OR
- R drops by > 5% from AV-34 baseline, OR
- P_lb or R_lb drops below gate floor
