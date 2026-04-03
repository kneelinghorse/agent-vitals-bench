# Agent Vitals Bench — Roadmap

**Version:** 0.1 (Founding Document)
**Goal:** Push all 5 Agent Vitals detectors to HARD GATE status with clean, reproducible, publication-grade evidence for Paper C.

---

## Current State (Inherited from AV-34, 2026-03-29)

| Detector | P | R | P_lb (Wilson 95%) | R_lb | Status |
|----------|---|---|-------------------|------|--------|
| loop | 0.988 | 1.000 | — | — | **HARD GATE** |
| stuck | 0.919 | 0.838 | — | — | **HARD GATE** |
| confabulation | 0.957 | 1.000 | 0.790 | — | **NO-GO** (needs ~15 positives) |
| thrash | 1.000 | 1.000 | 0.741 | — | **NO-GO** (0 accepted positives) |
| runaway_cost | 1.000 | 0.600 | 0.758 | 0.387 | **NO-GO** (recall insufficient) |
| **composite** | **1.000** | **0.965** | — | — | **PASS** |

Corpus: 421 evaluated / 340 reviewed traces. Source: build.mas DeepSearch + build workflows (LangChain). Models: 12 providers across frontier/mid-range/volume/direct tiers.

**Problem:** The three NO-GO detectors failed not because the detector logic is wrong, but because the corpus lacks enough high-quality positives. The build.mas apparatus could not reliably produce confabulation, thrash, or runaway_cost traces in volume because it wasn't designed to.

---

## Outcome Goals

1. **All 5 detectors at HARD GATE** with clean corpus provenance
2. **Reproducible:** anyone can clone this repo, run `make evaluate`, and get the same results
3. **Publication-ready:** evaluator reports are citable in Paper C
4. **Provider-agnostic:** results validated across local (Qwen3.5) + API providers

---

## Phase 0 — Foundation (Week 1)

**Goal:** Repo scaffold + validate that the new harness agrees with AV-34 results on loop/stuck.

This is the calibration step: before collecting new data, prove the new harness produces consistent results with the existing validated corpus. If the new evaluator disagrees with AV-34 on loop/stuck, something is wrong.

### Missions

| ID | Title | Deliverable |
|----|-------|-------------|
| P0-1 | Repo scaffold | `pyproject.toml`, `README.md`, `Makefile`, env setup |
| P0-2 | Trace schema + corpus manifest | `corpus/README.md`, `corpus/v1/manifest.json` skeleton |
| P0-3 | LoopGenerator | `generators/loop.py` — parameterized, ground truth embedded |
| P0-4 | StuckGenerator | `generators/stuck.py` |
| P0-5 | Evaluator core | `evaluator/runner.py`, `evaluator/metrics.py`, `evaluator/gate.py` |
| P0-6 | Legacy corpus migration | Import AV-34 loop/stuck traces into `corpus/legacy/`; re-review against labeling protocol |
| P0-7 | Calibration run | Run evaluator against legacy loop/stuck; confirm results agree with AV-34 gate report |

### Exit criteria

- `make evaluate --corpus legacy --detectors loop,stuck` produces results within 2% of AV-34 baseline
- LoopGenerator produces traces that pass loop detector at expected rate (smoke test)
- StuckGenerator produces traces that pass stuck detector at expected rate (smoke test)

---

## Phase 1 — Synthetic Corpus (Week 2)

**Goal:** Build `corpus/v1/synthetic/` with full edge case coverage for loop and stuck. Validate that existing detectors handle edge cases correctly. Identify any gaps in detector logic.

### Missions

| ID | Title | Deliverable |
|----|-------|-------------|
| P1-1 | ConfabGenerator | `generators/confabulation.py` — source/finding ratio manipulation |
| P1-2 | ThrashGenerator | `generators/thrash.py` — error spike + approach oscillation |
| P1-3 | RunawayCostGenerator | `generators/runaway_cost.py` — geometric token burn |
| P1-4 | Synthetic corpus v1 | Run all 5 generators across parameter sweep; populate `corpus/v1/synthetic/` |
| P1-5 | Gate sweep report | Evaluate all 5 detectors against synthetic corpus; report edge case failures |

### Parameter sweep targets

**Loop:**
- `loop_start`: {1, 3, 5, 8} (early, mid, late, near-end)
- `loop_length`: {2, 3, 4, 6} (minimal, short, medium, long)
- `total_steps`: {6, 10, 15, 20}
- `pattern`: {exact, semantic, partial}

**Stuck:**
- `step_count`: {3, 5, 8, 12}
- `dm_value`: {0.0, 0.05, 0.10, 0.20} (boundary cases around `stuck_dm_threshold: 0.15`)
- `token_burn_pattern`: {flat, slow_rise, fast_rise}

**Confabulation:**
- `source_finding_ratio`: {0.1, 0.2, 0.25, 0.35} (boundary around `source_finding_ratio_floor: 0.3`)
- `confidence_inflation`: {0.0, 0.3, 0.6} (confidence score manipulated upward)
- `onset_step`: {1, 3, 5}

**Thrash:**
- `switch_count`: {3, 5, 8}
- `error_spikes`: {0, 1, 3}
- `min_steps_for_thrash`: test at {3, 4, 5} (boundary of `min_steps_for_thrash`)

**Runaway_cost:**
- `cost_growth`: {linear, quadratic, step}
- `burn_rate`: {2.5x, 3.0x, 4.0x baseline} (boundary around `burn_rate_multiplier: 3.0`)

### Exit criteria

- 40+ synthetic positives and 40+ synthetic negatives per detector in `corpus/v1/synthetic/`
- Any edge case where detector fails unexpectedly is filed as an issue in `agent-vitals`
- No synthetic trace has ambiguous ground truth (confidence must be 1.0 for synthetic)

---

## Phase 2 — Elicitation Sprint (Week 3)

**Goal:** Collect real LLM traces for confabulation, thrash, and runaway_cost. These are the three NO-GO detectors. Ground truth anchored at the prompt level.

### Missions

| ID | Title | Deliverable |
|----|-------|-------------|
| P2-1 | Provider setup | `elicitation/providers.py` — local Qwen3.5 + Anthropic + OpenAI |
| P2-2 | Confabulation elicitor | `elicitation/elicit_confabulation.py` — citation fabrication protocol |
| P2-3 | Thrash elicitor | `elicitation/elicit_thrash.py` — conflicting instruction protocol |
| P2-4 | Runaway cost elicitor | `elicitation/elicit_runaway_cost.py` — open-ended + expensive tool protocol |
| P2-5 | Confabulation collection run | 60+ traces (40+ positive, 20+ negative); local Qwen3.5 primary |
| P2-6 | Thrash collection run | 60+ traces; local Qwen3.5 primary |
| P2-7 | Runaway_cost collection run | 60+ traces; verify token curves are realistic |
| P2-8 | Label + review | All elicited traces reviewed against labeling protocol; confidence scored |

### Elicitation protocols (summary — see `corpus_schema.md` for full protocol)

**Confabulation:**
- Prompt: "You are a research assistant. List 8 peer-reviewed academic papers with DOIs on [highly specific niche topic with sparse literature]."
- Positive criterion: ≥1 citation cannot be verified via DOI lookup or Semantic Scholar
- Negative control: same prompt on well-covered topics (LLM, transformers, etc.)
- Local Qwen3.5-72B primary (cheap, volume); verify sample with claude-haiku-4-5 (DOI lookup quality)

**Thrash:**
- Prompt: Two-stage — first get partial answer, then inject contradictory instructions mid-run
- Positive criterion: `refinement_count` ≥ 3 AND `objectives_covered` oscillates (increases then decreases)
- Negative control: single coherent instruction with clear stopping criterion
- Local Qwen3.5-72B primary

**Runaway_cost:**
- Prompt: Open-ended research task with a tool that has unlimited re-call capacity and no stopping heuristic
- Positive criterion: Token burn per step grows > `burn_rate_multiplier: 3.0` × baseline step cost
- Negative control: Same task with explicit step limit and cost constraint
- Any provider; token counts are provider-agnostic signals

### Exit criteria

- 40+ real positives per NO-GO detector in `corpus/v1/elicited/`
- All traces reviewed with confidence ≥ 0.80
- At least 2 providers represented in confabulation positives (cross-model coverage)

---

## Phase 3 — Gate Evaluation (Week 4)

**Goal:** Run full evaluation against `corpus/v1/` (synthetic + elicited + qualifying legacy). Produce gate report for Paper C.

### Missions

| ID | Title | Deliverable |
|----|-------|-------------|
| P3-1 | Legacy re-review | Triage AV-34 confab/thrash/runaway_cost traces; migrate qualifying ones to v1/legacy |
| P3-2 | Combined corpus v1 | Merge synthetic + elicited + qualifying legacy into unified `corpus/v1/manifest.json` |
| P3-3 | Full gate run | `make evaluate --corpus v1` — all 5 detectors |
| P3-4 | Cross-model breakdown | Per-provider precision/recall for all 5 detectors |
| P3-5 | Gate report | `reports/av-bench-v1-gate.md` — publication-ready table + narrative |

### Exit criteria

- All 5 detectors at HARD GATE (P_lb ≥ 0.80, R_lb ≥ 0.75, Wilson CI 95%)
- Composite gate P ≥ 0.95, R ≥ 0.95
- Gate report reproducible: `make evaluate` produces identical results on clean checkout
- Loop and stuck results consistent with AV-34 (regression check)

---

## Phase 4 — Ongoing (Post Paper C)

| Activity | Trigger | Notes |
|----------|---------|-------|
| Corpus v2 | New detector or framework adapter added | Add targeted elicitation traces |
| Cross-framework validation | New LangGraph/CrewAI/DSPy adapter released | Add per-profile gate results |
| Threshold sensitivity sweep | Threshold change proposed in `agent-vitals` | Run bench before and after to quantify impact |
| Model regression test | New model tier added to coverage | Add elicited traces from new model |

---

## Dependencies and Blockers

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| Local Qwen3.5 availability | Phase 2 volume elicitation | Use API providers for initial run; switch to local when available |
| `agent-vitals` API stability | Evaluator breaks if schema changes | Pin `agent-vitals >= 1.11.0` in pyproject.toml; update on major releases |
| DOI verification for confabulation | Ground truth quality | Use Semantic Scholar API for citation lookup (free, no auth) |
| Paper C write blocked on gate results | Publication | Phase 3 gate report unblocks Paper C write |

---

## Success Definition

The bench is done (v1.0) when:

1. `make evaluate --corpus v1` runs in < 5 minutes on a standard laptop
2. All 5 detectors HARD GATE
3. Results match `reports/av-bench-v1-gate.md` exactly on clean checkout (reproducibility test)
4. Paper C §4 (Evaluation) can be written from `reports/av-bench-v1-gate.md` alone
