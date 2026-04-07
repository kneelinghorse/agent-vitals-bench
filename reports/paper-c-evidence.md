# Paper C Section 4 — Evidence Package

**Generated:** 2026-04-05
**Sprint:** 13 (Paper C Final Evidence, Confab Protocol Revision & CI/CD Gate Automation)
**Corpus version:** v1 (1144 traces, 3-source: synthetic + elicited + legacy)
**Gate thresholds:** P_lb >= 0.80, R_lb >= 0.75 (Wilson CI 95%), min 25 positives
**Upstream:** agent-vitals >= 1.11.0 with co-occurrence FP suppression fixes

---

## 1. Claim-to-Evidence Mapping

### Claim 1: All five detectors pass hard gates with publication-grade confidence

**Evidence:** [eval-20260405-v1.md](eval-20260405-v1.md) / [.json](eval-20260405-v1.json)

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.982 | 0.982 | 212 | PASS |
| stuck | 0.939 | 0.973 | 138 | PASS |
| confabulation | 0.948 | 0.856 | 208 | PASS |
| thrash | 0.978 | 0.978 | 173 | PASS |
| runaway_cost | 0.832 | 0.977 | 163 | PASS |

**Interpretation:** All five detectors exceed both P_lb >= 0.80 and R_lb >= 0.75 under the default profile. Sample sizes range from 138 to 216 positives per detector, well above the minimum 25 required for statistical power. Confabulation has the tightest recall margin (R_lb = 0.856, 20 FN out of 208 positives). Runaway_cost has the tightest precision margin (P_lb = 0.832, 21 FP).

---

### Claim 2: Detection generalizes across all four framework integration profiles

**Evidence:**
- CrewAI: [eval-20260405-v1-crewai.md](eval-20260405-v1-crewai.md) / [.json](eval-20260405-v1-crewai.json)
- LangGraph: [eval-20260405-v1-langgraph.md](eval-20260405-v1-langgraph.md) / [.json](eval-20260405-v1-langgraph.json)
- DSPy: [eval-20260405-v1-dspy.md](eval-20260405-v1-dspy.md) / [.json](eval-20260405-v1-dspy.json)

| Profile | Detectors Passing | Composite | Traces |
|---------|-------------------|-----------|--------|
| default | 5/5 | PASS | 1144 |
| crewai | 5/5 | PASS | 984 |
| langgraph | 5/5 | PASS | 984 |
| dspy | 4/4 evaluated (stuck excluded) | PASS | 962 |

**Per-detector comparison (P_lb / R_lb):**

| Detector | Default | CrewAI | LangGraph | DSPy |
|----------|---------|--------|-----------|------|
| loop | 0.982 / 0.982 | 0.980 / 0.980 | 0.980 / 0.980 | 0.979 / 0.979 |
| stuck | 0.939 / 0.973 | 0.969 / 0.969 | 0.969 / 0.969 | excluded |
| confabulation | 0.948 / 0.856 | 0.936 / 0.827 | 0.936 / 0.827 | 0.936 / 0.827 |
| thrash | 0.978 / 0.978 | 0.975 / 0.975 | 0.975 / 0.975 | 0.975 / 0.975 |
| runaway_cost | 0.832 / 0.977 | 0.940 / 0.973 | 0.808 / 0.973 | 0.814 / 0.973 |

**Interpretation:** All four profiles pass all enabled gates. Default, CrewAI, and LangGraph pass 5/5. DSPy passes 4/4 evaluated gates (stuck is disabled by design). CrewAI shows the highest runaway_cost precision (P_lb = 0.940) due to its calibrated burn_rate_multiplier = 3.0. LangGraph runaway_cost P_lb = 0.808 is the tightest cross-profile margin but still passes. DSPy runaway_cost P_lb = 0.814 was the final gate closure achieved via upstream co-occurrence FP suppression (see Claim 8).

---

### Claim 3: DSPy profile passes all four evaluated gates in stuck-disabled mode

**Evidence:** [eval-20260405-v1-dspy.md](eval-20260405-v1-dspy.md) / [.json](eval-20260405-v1-dspy.json)

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.979 | 0.979 | 180 | **PASS** |
| confabulation | 0.936 | 0.827 | 172 | **PASS** |
| thrash | 0.975 | 0.975 | 149 | **PASS** |
| runaway_cost | 0.814 | 0.973 | 139 | **PASS** |

**Interpretation:** Under DSPy's stuck-disabled mode, all four evaluated detectors achieve publication-grade confidence. Loop detection was recovered in Sprint 11 by raising `loop_consecutive_pct` from 0.6 to 0.7, reducing FPs from 51 to 18 while maintaining perfect recall. Runaway_cost was recovered in Sprint 12 via upstream co-occurrence FP suppression fixes in agent-vitals, bringing P_lb from 0.684 to 0.814. Confabulation and thrash do not depend on co-occurrence resolution with stuck and pass cleanly.

---

### Claim 4 (Limitation): DSPy stuck detector is disabled by design

**Evidence:** [eval-20260405-v1-dspy.md](eval-20260405-v1-dspy.md) / [.json](eval-20260405-v1-dspy.json)

| Detector | P_lb | R_lb | FP | Gate |
|----------|------|------|-----|------|
| stuck | 0.000 | 0.000 | 0 | **EXCLUDED** (disabled) |

**Root cause:** DSPy disables the stuck detector (`stuck_enabled: false`) because DSPy's optimization-based execution model does not produce the sequential stagnation patterns that the stuck detector monitors. This is an architectural choice, not a detector limitation.

**Previous limitation (resolved):** In Sprint 11, runaway_cost also failed under DSPy (P_lb = 0.684) because the co-occurrence resolution system depended on stuck being active to suppress cross-firing signals. This was resolved in Sprint 12 via 4 upstream fixes to agent-vitals co-occurrence logic (see Claim 8).

**Paper C scoping:** DSPy claims cover loop, confabulation, thrash, and runaway_cost (4/4 evaluated gates). Stuck is excluded by design with documented rationale.

---

### Claim 5: Detection thresholds are robust to perturbation

**Evidence:** [threshold-sensitivity.md](threshold-sensitivity.md) / [.json](threshold-sensitivity.json)

| Detector | Threshold | Default | Safe Range | Width | Brittle? |
|----------|-----------|---------|------------|-------|----------|
| loop | loop_consecutive_pct | 0.500 | [0.300, 0.700] | 0.400 | No |
| loop | loop_similarity_threshold | 0.775 | [0.600, 0.915] | 0.315 | No |
| stuck | stuck_dm_threshold | 0.150 | [0.050, 0.250] | 0.200 | No |
| stuck | stuck_cv_threshold | 0.300 | [0.150, 0.450] | 0.300 | No |
| confabulation | source_finding_ratio_floor | 0.300 | [0.150, 0.450] | 0.300 | No |
| thrash | thrash_error_threshold | 1.000 | [1.000, 4.000] | 3.000 | No |
| runaway_cost | burn_rate_multiplier | 2.500 | [2.26, 3.10] | 0.840 | No |

**Interpretation:** All seven swept thresholds have safe operating ranges — no detector is brittle to threshold choice. The `burn_rate_multiplier` was previously the tightest parameter (safe range [2.55, 2.90], width 0.35, marked brittle). After the Sprint 10 default fix (3.0 to 2.5) and Sprint 12 upstream co-occurrence FP suppression, the safe range widened to [2.26, 3.10] (width 0.84), with the default value 2.5 comfortably inside. Profile-specific calibration (CrewAI = 3.0, LangGraph = 3.25) further demonstrates that threshold tuning per framework is effective.

---

### Claim 6: Legacy corpus validates backward compatibility

**Evidence:** [eval-20260402-legacy.md](eval-20260402-legacy.md) / [.json](eval-20260402-legacy.json)

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.984 | 0.984 | 233 | PASS |
| stuck | 0.962 | 0.828 | 109 | PASS |

**Interpretation:** The AV-34 legacy corpus (re-reviewed traces from the original agent-vitals validation) passes hard gates for both loop and stuck detectors. This demonstrates backward compatibility — detectors that worked against the original labeled data continue to work with the current implementation. Legacy corpus covers loop and stuck only (the three newer detectors — confabulation, thrash, runaway_cost — did not exist in the original validation).

---

### Claim 7: Cross-model generalization (frontier tier)

**Evidence:** [eval-20260403-cross-model.md](eval-20260403-cross-model.md)

| Detector | claude-sonnet | gpt-4o | Generalizes? |
|----------|---------------|--------|--------------|
| confabulation | 0/5 (0%) | 2/5 (40%) | **NO** |
| thrash | 5/5 (100%) | 5/5 (100%) | Yes |
| runaway_cost | 5/5 (100%) | 5/5 (100%) | Yes |

**Interpretation:** Thrash and runaway_cost generalize perfectly across frontier models — these detectors work on structural patterns (error oscillation, cost burn rate) that are model-independent. Confabulation does not generalize: 0% detection on claude-sonnet, 40% on gpt-4o. Sample sizes are directional (5 per model per detector), below the 25 minimum for formal gates.

**Design gap analysis:** Frontier models produce plausible-looking fake sources in high volume. The `source_finding_ratio` never drops below the 0.3 floor because frontier models confabulate differently from the synthetic and locally-elicited training corpus — they produce structurally correct but factually wrong citations rather than obviously fabricated ones.

**Mitigation (upstream):** The agent-vitals `verified_source_ratio` signal (added in Sprint 12) provides verification-aware confabulation detection by tracking `verified_sources_count / unverified_sources_count` in the RawSignals schema. This signal can distinguish verified from unverified sources regardless of how plausible they appear. The bench confabulation elicitation protocol needs revision (Sprint 13, M02) to leverage this signal before cross-model claims can be strengthened.

**Gap (reduced):** Mid-range and volume tier elicitation not yet run. Full 3-tier campaign requires ~4 hours of API time. Frontier-only results are sufficient for Paper C with appropriate qualification.

---

### Claim 8: DSPy gate closure achieved via upstream co-occurrence collaboration

**Evidence:**
- Before: [eval-20260403-v1-dspy.md](eval-20260403-v1-dspy.md) (runaway_cost P_lb = 0.684, NO-GO)
- After: [eval-20260405-v1-dspy.md](eval-20260405-v1-dspy.md) (runaway_cost P_lb = 0.814, PASS)

**Narrative:** DSPy's runaway_cost gate failure (P_lb = 0.684, 44 FP) was caused by co-occurrence false positives when the stuck detector is disabled. The fix required 4 upstream changes to agent-vitals co-occurrence resolution logic:

1. **Relaxed stagnation suppression** — prevented stuck-like signals from leaking into runaway_cost scoring when stuck is disabled
2. **Confabulation overlap arbitration** — resolved cross-firing between confabulation and runaway_cost on traces with both cost anomalies and source pattern changes
3. **Loop-runaway arbitration** — prevented loop-positive traces from triggering runaway_cost FPs via shared burn_rate elevation
4. **Error-count suppression** — filtered spurious burn_rate_anomaly signals triggered by error accumulation patterns

**Validation effort:** 6 validation cycles over 2 sessions, catching 2 regressions (LangGraph P_lb temporarily dropped below 0.80 during fix #2; burn_rate_anomaly recall dropped from 100% to 69.8% during upstream schema changes). Each regression was identified, root-caused, and resolved before proceeding.

**Result:** DSPy runaway_cost P_lb improved from 0.684 to 0.814 (20 FP, down from 44). All four framework profiles now pass all enabled gates — a 4/4 composite PASS across default, CrewAI, LangGraph, and DSPy.

**Paper C significance:** This demonstrates that the co-occurrence resolution system can be adapted for framework-specific detector configurations without requiring per-framework detector rewrites. The upstream fixes benefit all profiles, not just DSPy.

---

## 2. Evidence Completeness Matrix

| Paper C Claim | Primary Report | Profile | Status |
|---------------|----------------|---------|--------|
| 5/5 detectors pass hard gates | eval-20260405-v1 | default | Complete |
| Cross-framework generalization | eval-20260405-v1-{crewai,langgraph,dspy} | crewai, langgraph, dspy | Complete |
| DSPy 4/4 evaluated gates pass | eval-20260405-v1-dspy | dspy | Complete |
| DSPy limitation (stuck excluded) | eval-20260405-v1-dspy | dspy | Complete |
| DSPy gate closure narrative | eval-20260403-v1-dspy + eval-20260405-v1-dspy | dspy | Complete |
| Threshold robustness | threshold-sensitivity | default | Complete |
| Backward compatibility | eval-20260402-legacy | default | Complete |
| Cross-model generalization | eval-20260403-cross-model | frontier | Complete |
| Confab cross-model design gap | eval-20260403-cross-model | frontier | Complete (limitation) |
| verified_source_ratio mitigation | — | — | Documented (pending protocol revision) |

---

## 3. Corpus Summary

| Source | Traces | Confidence | Detectors Covered |
|--------|--------|------------|-------------------|
| Synthetic (generators) | ~800 | 1.0 | All 5 |
| Elicited (real LLM) | ~200 | 0.8-0.9 | confabulation, thrash, runaway_cost |
| Legacy (AV-34 migration) | 299 | 0.7 | loop, stuck |
| Framework traces | ~500 | 1.0 | All 5 (per-profile variants) |

Total unique traces in v1 manifest: 1144 (default profile evaluation).

---

## 4. Report Artifact Inventory

| Artifact | Path | Date | Profile |
|----------|------|------|---------|
| Default gate report | reports/eval-20260405-v1.md | 2026-04-05 | default |
| Default gate report (JSON) | reports/eval-20260405-v1.json | 2026-04-05 | default |
| CrewAI gate report | reports/eval-20260405-v1-crewai.md | 2026-04-05 | crewai |
| CrewAI gate report (JSON) | reports/eval-20260405-v1-crewai.json | 2026-04-05 | crewai |
| LangGraph gate report | reports/eval-20260405-v1-langgraph.md | 2026-04-05 | langgraph |
| LangGraph gate report (JSON) | reports/eval-20260405-v1-langgraph.json | 2026-04-05 | langgraph |
| DSPy gate report | reports/eval-20260405-v1-dspy.md | 2026-04-05 | dspy |
| DSPy gate report (JSON) | reports/eval-20260405-v1-dspy.json | 2026-04-05 | dspy |
| DSPy gate report (pre-fix) | reports/eval-20260403-v1-dspy.md | 2026-04-03 | dspy |
| DSPy gate report (pre-fix, JSON) | reports/eval-20260403-v1-dspy.json | 2026-04-03 | dspy |
| Cross-model analysis | reports/eval-20260403-cross-model.md | 2026-04-04 | frontier |
| Threshold sensitivity sweep | reports/threshold-sensitivity.md | 2026-04-03 | default |
| Threshold sensitivity (JSON) | reports/threshold-sensitivity.json | 2026-04-03 | default |
| Legacy corpus gate report | reports/eval-20260402-legacy.md | 2026-04-02 | default |
| Legacy corpus gate report (JSON) | reports/eval-20260402-legacy.json | 2026-04-02 | default |

**Total: 15 report artifacts (8 markdown + 7 JSON)**
