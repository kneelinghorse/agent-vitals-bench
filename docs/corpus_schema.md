# Corpus Schema and Labeling Protocol

**Version:** 0.1
**Scope:** All traces in `corpus/` regardless of tier

---

## 1. Trace File Format

Traces are stored as **JSON arrays** of `VitalsSnapshot` objects (schema from `agent-vitals/agent_vitals/schema.py`).

```json
[
  {
    "spec_version": "1.0.0",
    "timestamp": "2026-01-01T00:00:00Z",
    "mission_id": "bench-loop-001",
    "run_id": "loop-syn-001",
    "loop_index": 0,
    "signals": {
      "findings_count": 3,
      "sources_count": 5,
      "objectives_covered": 1,
      "coverage_score": 0.45,
      "confidence_score": 0.6,
      "prompt_tokens": 200,
      "completion_tokens": 180,
      "total_tokens": 380,
      "api_calls": 1,
      "query_count": 2,
      "unique_domains": 3,
      "refinement_count": 0,
      "convergence_delta": 0.12,
      "error_count": 0
    },
    "metrics": {
      "cv_coverage": 0.0,
      "cv_findings_rate": 0.0,
      "dm_coverage": 0.5,
      "dm_findings": 0.5,
      "qpf_tokens": 1.0,
      "cs_effort": 0.5
    },
    "health_state": "healthy",
    "health_state_changed": false,
    "previous_health_state": null,
    "loop_detected": false,
    "loop_confidence": 0.0,
    "loop_trigger": null,
    "stuck_detected": false,
    "stuck_confidence": 0.0,
    "stuck_trigger": null,
    "confabulation_detected": false,
    "confabulation_confidence": 0.0,
    "confabulation_trigger": null,
    "confabulation_signals": [],
    "thrash_detected": false,
    "runaway_cost_detected": false,
    "intervention": null
  }
]
```

**Note:** The `loop_detected`, `stuck_detected`, etc. fields in the stored trace reflect what the detector computed *during generation*. For synthetic traces, these may all be `false` in the stored snapshots — the evaluator re-runs detection against the full snapshot sequence during evaluation.

---

## 2. Manifest Entry Schema

Every trace has exactly one entry in `corpus/vN/manifest.json`:

```json
{
  "trace_id": "loop-syn-001",
  "path": "traces/loop/positive/loop-syn-001.json",
  "tier": "synthetic",
  "labels": {
    "loop": true,
    "stuck": false,
    "confabulation": false,
    "thrash": false,
    "runaway_cost": false
  },
  "metadata": {
    "generator": "LoopGenerator",
    "params": {
      "loop_start": 3,
      "loop_length": 4,
      "total_steps": 12,
      "pattern": "exact"
    },
    "onset_step": 3,
    "tier": "synthetic",
    "model": null,
    "provider": null,
    "framework": null,
    "reviewer": null,
    "review_date": null,
    "confidence": 1.0,
    "notes": ""
  }
}
```

**Required fields:** `trace_id`, `path`, `tier`, `labels`, `metadata.confidence`

**Confidence scoring:**

| Value | Meaning |
|-------|---------|
| 1.0 | Synthetic — ground truth is certain (generated) |
| 0.9 | Elicited — failure mode unambiguously triggered by prompt design; verified by signal check |
| 0.8 | Elicited — failure mode triggered; minor ambiguity in onset step |
| 0.7 | Legacy — reviewed by original corpus team; meets current labeling criteria |
| < 0.7 | Do not include in gate evaluation corpus |

**The evaluator filters on confidence ≥ 0.8 by default.** Traces with confidence < 0.8 are stored but excluded from precision/recall calculation.

---

## 3. Labeling Protocol

### 3.1 Loop

**Positive label criteria (ALL must hold):**
- The trace contains a segment where `findings_count` does not increase for ≥ `findings_plateau_pct × trace_length` consecutive steps
- OR `coverage_score` changes by < 0.01 across ≥ `loop_consecutive_pct × trace_length` steps
- OR (if `output_text` present) Jaccard similarity between consecutive outputs ≥ `loop_similarity_threshold: 0.8`
- Trace length ≥ `min_evidence_steps: 3`

**Onset step:** First step where the plateau or similarity condition is satisfied.

**Negative label criteria:**
- Progress is visible throughout (findings, coverage both grow)
- OR trace is too short (< `min_evidence_steps`)
- OR apparent repetition is due to framework mechanics (e.g., DSPy optimization loop — label with `framework: dspy`)

**Common false positive risks:**
- Research tasks with genuinely limited source material (correct to have flat findings, not a loop)
- LangGraph checkpointing replays that look like repeated steps

### 3.2 Stuck

**Positive label criteria (ALL must hold):**
- `dm_coverage` ≤ `stuck_dm_threshold: 0.15` for ≥ 3 consecutive steps
- AND `cv_coverage` ≤ `stuck_cv_threshold: 0.3`
- AND `total_tokens` is increasing (burning tokens despite no progress)
- AND `source_productive` is False (see §3.6 for definition)
- Workflow is `research-only` (or `workflow_stuck_enabled: all` in profile)

**Onset step:** First step where all three conditions are satisfied simultaneously.

**Negative label criteria:**
- DM is low but tokens are flat (agent has paused, not burned)
- Build workflows (stuck detection disabled for build by default profile)
- `source_productive` is True — sufficient sources, findings, and coverage indicate
  productive research, not stagnation

### 3.3 Confabulation

**Positive label criteria:**
- `sources_count / findings_count` < `source_finding_ratio_floor: 0.3` AND this ratio has been declining for ≥ `source_finding_ratio_declining_steps: 3` consecutive steps
- OR `confidence_score` ≥ 0.85 while `sources_count` = 0 (high confidence with no sources)
- For elicited traces: ≥ 1 citation in the agent's output cannot be verified via Semantic Scholar DOI lookup

**Onset step:** First step where the ratio floor is breached.

**Negative label criteria:**
- Low source count is legitimate (early in research, no sources found yet) — onset should be mid/late trace
- High confidence with no sources on tasks where external sources are not expected (e.g., math problems)

**Verification for elicited traces:**
```python
# Use Semantic Scholar API (free, no auth)
GET https://api.semanticscholar.org/graph/v1/paper/{doi}
# If 404 → citation is unverifiable → positive label
```

### 3.4 Thrash

**Positive label criteria:**
- `error_count` spikes ≥ `thrash_error_threshold: 1` AND
- `refinement_count` increases AND
- `objectives_covered` oscillates (increases then decreases at least once)
- Trace length ≥ `min_steps_for_thrash: 3`

**Onset step:** First step where error spike + approach change co-occur.

**Negative label criteria:**
- Errors occur but agent corrects and continues without reversing direction
- Multiple objectives covered monotonically (that's good coverage, not thrash)

**Common false positive risks:**
- Multi-agent handoffs that look like approach switching
- LangGraph state machine transitions that register as objective changes

### 3.5 Runaway Cost

**Positive label criteria:**
- Token cost per step is increasing AND
- The per-step cost at any point exceeds `burn_rate_multiplier: 3.0` × average of first 3 steps
- OR the stuck trigger fires as `burn_rate_anomaly`

**Onset step:** First step where burn rate multiplier is exceeded.

**Negative label criteria:**
- Large single step with normal subsequent steps (spike, not runaway)
- Token counts high but flat (expensive-but-stable, not runaway)

### 3.6 Source Productive Gate

The `source_productive` condition suppresses stuck, loop, and token_usage_variance_flat
triggers when the agent is demonstrably finding useful sources. It requires ALL THREE
conditions simultaneously:

| Condition | Threshold | Constant |
|-----------|-----------|----------|
| `sources_count` ≥ 10 | `SOURCE_PRODUCTIVITY_MIN_SOURCES` | `loop.py:26` |
| `findings_count` ≥ 5 | `SOURCE_PRODUCTIVITY_MIN_FINDINGS` | `loop.py:27` |
| `coverage_score` ≥ 0.5 | hardcoded in gate | `loop.py:173` |

**Why coverage ≥ 0.5?** Without this condition, an agent with many sources and findings
but low coverage (e.g., collecting citations without synthesizing) could suppress stuck
detection despite making no real progress. The coverage floor ensures the sources are
translating into actual task coverage.

**Impact on generators:** The RunawayCostGenerator explicitly maintains `source_productive=True`
during runaway steps (coverage floored at 0.5, sources ≥ 10, findings ≥ 5) to prevent
stuck cross-triggers. Without this, token_usage_variance_flat fires on runaway traces
because the flat variance looks like stuck behavior.

### 3.7 Signal Mapping (Model Size Awareness)

Small models (4B/9B) produce structurally flat token variance regardless of `max_tokens`.
The signal mapping module (`detection/signal_mapping.py`) auto-classifies model size from
the coefficient of variation (CV) of `completion_tokens` across the trace:

| Model Class | CV Threshold | Adjustments |
|-------------|-------------|-------------|
| Small | CV < 0.15 | Suppress `token_usage_variance_flat`; scale `burn_rate_multiplier` × 2.0 |
| Medium | 0.15 ≤ CV < 0.30 | Scale `burn_rate_multiplier` × 1.5 |
| Large | CV ≥ 0.30 | No adjustments (default behavior) |

Configurable via `model_size_class` in VitalsConfig (`"auto"`, `"small"`, `"medium"`, `"large"`).
Default `"auto"` requires ≥ 4 steps with non-zero `completion_tokens` for classification.

### 3.8 Co-Occurrence Resolution Rules

When multiple detectors fire on the same trace, the evaluator applies these resolution
rules (see `evaluator/runner.py:147-201`):

**Stuck suppression:**
- Stuck is only counted when `detector_priority != "confabulation"` AND
  `stuck_trigger != "burn_rate_anomaly"` (line 148-153)
- Rationale: confabulation takes priority over stuck; burn_rate_anomaly maps to
  runaway_cost, not stuck

**Runaway cost bridge:**
- `stuck_trigger == "burn_rate_anomaly"` maps to `runaway_cost_detected` (line 169-170)
- Only counts when `not stuck_fired` — if stuck is the root cause, token growth is a
  symptom of being stuck, not independent runaway (line 184)

**Loop + stuck co-occurrence:**
- When both loop and stuck fire (and neither thrash nor runaway fires), stuck is
  suppressed IF loop has `content_similarity` evidence AND `preserve_stuck_overlap`
  is not set (lines 190-195)
- `preserve_stuck_overlap` is set when any step has `detector_priority == "stuck"`,
  indicating stuck is the primary failure mode (lines 139-142)

**Thrash minimum steps:**
- Thrash requires `step_count >= min_steps_for_thrash` (default 3) to avoid
  segmentation artifacts on very short traces (stop_rule.py:84-91)

---

## 4. Corpus Directory Structure

```
corpus/
├── README.md                           (this document summary + schema)
├── legacy/
│   ├── manifest.json                   (AV-34 traces, re-reviewed)
│   └── traces/
│       ├── loop/positive/
│       ├── loop/negative/
│       ├── stuck/positive/
│       └── stuck/negative/
│           (confab/thrash/runaway_cost demoted — re-review pending)
└── v1/
    ├── manifest.json                   (all v1 traces: synthetic + elicited + qualifying legacy)
    └── traces/
        ├── loop/
        │   ├── positive/
        │   └── negative/
        ├── stuck/
        │   ├── positive/
        │   └── negative/
        ├── confabulation/
        │   ├── positive/
        │   └── negative/
        ├── thrash/
        │   ├── positive/
        │   └── negative/
        └── runaway_cost/
            ├── positive/
            └── negative/
```

**File naming convention:** `{detector}-{tier_prefix}-{sequence}.json`
- `loop-syn-001.json` — loop, synthetic, sequence 001
- `confab-eli-012.json` — confabulation, elicited, sequence 012
- `stuck-leg-003.json` — stuck, legacy, sequence 003

---

## 5. Multi-Label Traces

A trace may trigger multiple detectors. This is expected and should be labeled correctly:

```json
"labels": {
  "loop": true,
  "stuck": true,
  "confabulation": false,
  "thrash": false,
  "runaway_cost": false
}
```

A loop that also exhibits stuck behavior (DM near zero, burning tokens) is a valid multi-label trace. Store it in `traces/loop/positive/` with the loop as primary label; the evaluator will read all labels from the manifest.

---

## 6. Minimum Corpus Size for Gate Promotion

Wilson CI math at 95% confidence:

| Target P_lb | Observed precision needed | Required positives |
|------------|--------------------------|-------------------|
| 0.80 | 0.90 | ~35 |
| 0.80 | 0.95 | ~25 |
| 0.75 | 0.85 | ~40 |

**Rule of thumb:** Target **40 positives and 40 negatives per detector** in `corpus/v1/`. This provides comfortable margin above the Wilson CI threshold even with a few unexpected false positives/negatives.

---

## 7. Legacy Corpus Triage Guide

When migrating AV-34 traces into `corpus/legacy/`:

**Accept (carry forward):**
- Confidence ≥ 0.70
- Loop/stuck traces: meets current labeling criteria above
- Multi-step traces (≥ 5 steps) with clear signal patterns

**Demote (move to `corpus/legacy/demoted/`):**
- Traces with < `min_evidence_steps: 3` steps
- Confab/thrash/runaway_cost traces from AV-34 — these were collected from build.mas and may not meet the new labeling criteria
- Any trace where the reviewer noted "mathematically simple" or "forced"
- Traces where onset is ambiguous (reviewer disagreement or no onset_step recorded)

**The loop and stuck hard gate results from AV-34 stand.** Demoting legacy traces does not invalidate the AV-34 gate report — it just means the new bench evaluation starts from a clean corpus and should reproduce those results independently.

---

## 8. Generator Threshold Reference

Generators produce synthetic traces with hardcoded thresholds that must align with
detector gate conditions. This section documents all thresholds by generator.

### 8.1 Detection Gate Thresholds (from `agent_vitals/detection/`)

These are the primary thresholds generators must respect:

| Parameter | Default | Source | Used by |
|-----------|---------|--------|---------|
| `loop_similarity_threshold` | 0.8 | config.py | Loop (content_similarity) |
| `loop_consecutive_pct` | 0.5 | config.py | Loop (plateau window) |
| `findings_plateau_pct` | 0.4 | config.py | Loop/stuck (findings_plateau) |
| `min_evidence_steps` | 3 | config.py | All detectors |
| `source_finding_ratio_floor` | 0.3 | config.py | Confabulation |
| `source_finding_ratio_declining_steps` | 3 | config.py | Confabulation |
| `stuck_dm_threshold` | 0.15 | config.py | Stuck (coverage_stagnation) |
| `stuck_cv_threshold` | 0.3 | config.py | Stuck (coverage_stagnation) |
| `burn_rate_multiplier` | 3.0 | config.py | Runaway cost |
| `token_scale_factor` | 1.0 | config.py | Runaway cost (scales tokens) |
| `thrash_error_threshold` | 1 | stop_rule.py | Thrash |
| `min_steps_for_thrash` | 3 | stop_rule.py | Thrash (short segment guard) |
| `SOURCE_PRODUCTIVITY_MIN_SOURCES` | 10 | loop.py | source_productive gate |
| `SOURCE_PRODUCTIVITY_MIN_FINDINGS` | 5 | loop.py | source_productive gate |
| coverage floor in source_productive | 0.5 | loop.py:173 | source_productive gate |

### 8.2 Per-Generator Thresholds

**LoopGenerator** (`generators/loop.py`):
- Output similarity in loop phase: 0.92 (exceeds `loop_similarity_threshold: 0.8`)
- Pattern-specific `cv_coverage`: 0.32 (exact), 0.35 (semantic/partial)
- `dm_coverage` in loop phase: 0.12 (deliberately below `stuck_dm_threshold: 0.15`)
- `dm_findings` in loop phase: 0.05

**StuckGenerator** (`generators/stuck.py`):
- Output similarity in stuck phase: 0.15 (low — stuck agents don't repeat)
- Finding growth rate: `steps_stuck // 3` (very slow progress)
- Coverage scaling factor: 0.85 with 0.005 increment per stuck step
- Token burn patterns: "flat" (constant), "slow_rise" (linear growth), "fast_rise" (2x per step)

**ConfabGenerator** (`generators/confabulation.py`):
- Healthy confidence baseline: 0.3
- Sources in healthy phase: `findings + 3` (safely above ratio floor)
- Frozen sources at onset: `onset_step + 5`
- Convergence delta in confab phase: 0.02 (low but non-zero)

**ThrashGenerator** (`generators/thrash.py`):
- Error accumulation per thrash step: distributed via `errors_per_step`
- Finding growth rate: `steps_thrash // 3` (slow, like stuck)
- `cv_coverage` base: 0.5, increment: +0.05/step (high variance = instability)
- `dm_coverage`: 0.2 (just above `stuck_dm_threshold: 0.15`)
- Convergence delta: 0.01 (near-zero)

**RunawayCostGenerator** (`generators/runaway_cost.py`):
- Token burn curves start at 80% of `burn_rate` (floor to ensure first runaway step
  exceeds detection threshold — without this, gradual curves never trigger because
  growing deltas raise the baseline average)
- Coverage floor: 0.5 (maintains `source_productive=True`)
- `cv_coverage`: 0.35 (above `stuck_cv_threshold: 0.3`)
- `dm_coverage`: 0.25 (above `stuck_dm_threshold: 0.15`)
- `qpf_tokens`: 0.3 + 0.05/step (growing prompt-to-total ratio = unfairness signal)
