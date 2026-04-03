# Agent Vitals Bench — Technical Architecture

**Version:** 0.1 (Founding Document)
**Status:** Pre-build — use as CMOS initialization foundation
**Sibling repo:** `../agent-vitals` (`pip install agent-vitals`, v1.11.0+)
**Parent portfolio:** `metrics-protocols/` (alongside `metrics_and_protocols`, `build.mas`)

---

## 1. Purpose and Context

`agent-vitals-bench` is the **validation and benchmarking harness** for the Agent Vitals detector package. It is *not* a production tool — it is the apparatus that proves the detectors work with rigor sufficient for scientific publication (Paper C: Agent Vitals Empirical Paper).

### Why this exists

The previous test apparatus (build.mas) was a Langchain-based DeepSearch workflow cloned from the TraceLab research pipeline. It served as the first real-world harness for developing the Agent Vitals detectors and produced the AV-34 corpus (421 evaluated / 340 reviewed traces). However:

- **No controlled ground truth at generation time.** Traces were labeled retrospectively by reviewing production runs. Ground truth was inferred, not designed.
- **No parameter control.** Trace characteristics (loop length, onset step, severity) were whatever the workflow produced. Edge case coverage was accidental.
- **Two-workflow conflation.** Build.mas runs both a deep-research workflow and a build workflow. The corpus mixes these without clean provenance separation.
- **Mathematical simplicity risk.** Some traces — particularly in the corpus expansion phase — may have been accepted despite being structurally simple (low step counts, obvious signals). The "forced step 2" problem.
- **App-specific coupling.** All corpus traces are build.mas traces. The detectors have not been validated against a diverse trace population beyond that one application.

Agent Vitals is now a standalone pip package (`kneelinghorse/agent-vitals`). The bench is its sibling validation apparatus, independent of any production workflow.

---

## 2. Relationship to Agent Vitals

```
metrics-protocols/
├── agent-vitals/          # pip package — detectors, exporters, schema
│   ├── agent_vitals/
│   │   ├── detection/     # loop.py, similarity.py, stop_rule.py, cusum.py
│   │   ├── schema.py      # VitalsSnapshot, RawSignals, TemporalMetricsResult
│   │   ├── backtest.py    # ConfusionCounts, run_backtest()
│   │   ├── ci_gate.py     # wilson_interval(), evaluate_promotion()
│   │   └── thresholds.yaml
│   └── tests/             # Unit tests + smoke tests (stay in agent-vitals)
│
└── agent-vitals-bench/    # THIS REPO — corpus, generators, evaluator
    ├── docs/              # Foundational docs (this file)
    ├── generators/        # Synthetic trace factories
    ├── corpus/            # Versioned labeled trace corpus
    ├── evaluator/         # Runner, metrics, gate enforcement, reporter
    └── elicitation/       # Scripts to elicit real failure modes from LLMs
```

**Dependency direction:** `agent-vitals-bench` depends on `agent-vitals`. Not the other way around.

**Division of responsibility:**

| Concern | Lives in |
|---------|----------|
| Detector logic | `agent-vitals/agent_vitals/detection/` |
| Detector unit tests | `agent-vitals/tests/` |
| Trace corpus (labeled data) | `agent-vitals-bench/corpus/` |
| Synthetic trace generation | `agent-vitals-bench/generators/` |
| Precision/recall evaluation | `agent-vitals-bench/evaluator/` |
| Gate enforcement (promotion) | `agent-vitals-bench/evaluator/gate.py` |
| LLM elicitation harness | `agent-vitals-bench/elicitation/` |
| Reports / Paper C evidence | `agent-vitals-bench/reports/` |

---

## 3. Core Architecture

### 3.1 Three-Tier Corpus

Every trace in the corpus belongs to exactly one tier.

| Tier | Source | Ground truth | Use |
|------|--------|-------------|-----|
| **Synthetic** | Generators (this repo) | Perfect — failure mode is designed in at generation time | Edge case coverage, parameter sweeps, regression tests |
| **Elicited** | Real LLM runs via `elicitation/` scripts | High — failure mode is intentionally triggered by prompt design | Covers confabulation, thrash, runaway_cost gaps |
| **Legacy** | Existing build.mas / AV-34 corpus | Variable — requires re-review against current labeling protocol | Loop/stuck baselines; confab/thrash/runaway_cost demoted pending re-review |

The **key insight:** ground truth comes from *prompt design* or *generator parameters*, not from post-hoc reviewer labeling. This is the fix for the build.mas provenance problem.

### 3.2 Five Detectors

| Detector | Current status (AV-34) | Gate criteria | Notes |
|----------|----------------------|---------------|-------|
| `loop` | P=0.988, R=1.000, **HARD GATE** | P_lb ≥ 0.80, R_lb ≥ 0.75 (Wilson CI) | ✓ Proven |
| `stuck` | P=0.919, R=0.838, **HARD GATE** | P_lb ≥ 0.80, R_lb ≥ 0.75 (Wilson CI) | ✓ Proven |
| `confabulation` | P=0.957, R=1.000, P_lb=0.790, **NO-GO** | P_lb ≥ 0.80 | ~15 more positives needed |
| `thrash` | P=1.000, R=1.000, P_lb=0.741, **NO-GO** | P_lb ≥ 0.80 | 0 accepted positives in corpus |
| `runaway_cost` | P=1.000, R=0.600, P_lb=0.758, R_lb=0.387, **NO-GO** | P_lb ≥ 0.80, R_lb ≥ 0.75 | Precision ok, recall insufficient |

All gate criteria use **Wilson score confidence interval lower bounds at 95% confidence**. The implementation is in `agent-vitals/agent_vitals/ci_gate.py`:
```python
wilson_interval(successes=tp, trials=tp+fp)  # precision CI
wilson_interval(successes=tp, trials=tp+fn)  # recall CI
```

The bench's `evaluator/gate.py` reuses this exact function — no custom reimplementation.

### 3.3 Trace Schema

Traces in the corpus are sequences of `VitalsSnapshot` objects (defined in `agent-vitals/agent_vitals/schema.py`). A trace is a list of snapshots, one per agent loop step.

**Minimum viable snapshot (4 required fields in `RawSignals`):**
```python
RawSignals(
    findings_count=...,    # int ≥ 0
    coverage_score=...,    # float [0,1]
    total_tokens=...,      # int ≥ 0
    error_count=...,       # int ≥ 0
)
```

**Full signal surface (all fields):**
```
findings_count, sources_count, objectives_covered, coverage_score,
confidence_score, prompt_tokens, completion_tokens, total_tokens,
api_calls, query_count, unique_domains, refinement_count,
convergence_delta, error_count
```

**Computed temporal metrics (TemporalMetricsResult — populated by agent-vitals):**
```
cv_coverage, cv_findings_rate,
dm_coverage, dm_findings,       # Directional Momentum (USS Temporal dimension)
qpf_tokens,                     # Queue Position Fairness
cs_effort                       # Crescendo Symmetry
```

Trace files in the corpus are stored as **JSONL** (one snapshot per line) or **JSON array**. Both formats are accepted by the evaluator runner.

### 3.4 Corpus Manifest

Every trace in the corpus has a manifest entry in `corpus/vN/manifest.json`:

```json
{
  "trace_id": "loop-syn-001",
  "path": "traces/loop/positive/loop-syn-001.jsonl",
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
    "params": {"loop_start": 3, "loop_length": 4, "total_steps": 12, "pattern": "exact"},
    "onset_step": 3,
    "source": "synthetic",
    "model": null,
    "reviewer": null,
    "confidence": 1.0,
    "notes": ""
  }
}
```

For elicited traces, `generator` is null and `model`, `reviewer`, `confidence` are populated. For legacy traces, `tier` is `"legacy"` and `confidence` reflects reviewer certainty (0.0–1.0).

### 3.5 Generator Architecture

Each generator is a Python class that produces a `(trace, metadata)` pair with known ground truth.

```python
class LoopGenerator(TraceGenerator):
    def generate(
        self,
        loop_start: int,        # Step index where loop begins (0-indexed)
        loop_length: int,       # Number of steps in the loop
        total_steps: int,       # Total trace length
        pattern: str,           # "exact" | "semantic" | "partial"
        model: str = "synthetic",
    ) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        ...
```

The generator sets `findings_count`, `sources_count`, `coverage_score`, etc. to values that *structurally represent* the failure mode — without LLM content. The detector only sees the numerical signals, not text content (except for similarity-based loop detection which requires `output_text`).

**Generator responsibility matrix:**

| Generator | Key signals it manipulates | Simulated failure |
|-----------|---------------------------|------------------|
| `LoopGenerator` | `findings_count` plateau, `coverage_score` flatline, optionally `output_text` similarity | Repetitive steps, no progress |
| `StuckGenerator` | `dm_coverage` → 0, `cv_coverage` → 0, token burn with zero `convergence_delta` | Progress stopped, tokens burning |
| `ThrashGenerator` | `error_count` spikes, approach switches (via `objectives_covered` oscillation), high `refinement_count` | Back-and-forth without settling |
| `ConfabGenerator` | `sources_count` / `findings_count` ratio collapse (< `source_finding_ratio_floor: 0.3`), `confidence_score` inflation | Findings inflation, source-finding decoupling |
| `RunawayCostGenerator` | `total_tokens` per step grows geometrically, `burn_rate_anomaly` in stuck trigger | Token cost runaway |

### 3.6 Elicitation Harness

For detectors that require real LLM behavior (confabulation especially), synthetic signal manipulation is insufficient — the confabulation detector uses the `source_finding_ratio` and confidence signals, which are easy to fake, but production validity requires real LLM traces where the model *actually* fabricated.

The elicitation harness (`elicitation/`) runs purpose-built prompts designed to reliably trigger each failure mode:

**LLM providers available:**
- Local: Qwen3.5 series (up to 128GB) — primary for high-volume elicitation
- API: Anthropic (Claude), OpenAI (GPT-4o), configured via env vars

**Provider configuration:**
```python
# elicitation/providers.py
PROVIDERS = {
    "local-qwen": {"base_url": "http://localhost:11434/v1", "model": "qwen3.5:72b"},
    "anthropic":  {"model": "claude-haiku-4-5-20251001"},  # cheapest Anthropic tier
    "openai":     {"model": "gpt-4o-mini"},                # cheapest OpenAI tier
}
```

**Elicitation strategy per detector:**

| Detector | Elicitation method | Ground truth anchor |
|----------|-------------------|---------------------|
| `loop` | Prompt: "Continue researching X until you have 20 findings" with no new information available | Loop onset = first repeated step (detected by output similarity) |
| `stuck` | Prompt: "Solve Y" with tool results always returning empty/null | Stuck onset = after min_evidence_steps with zero convergence |
| `confabulation` | Prompt: "List 10 peer-reviewed papers on [obscure niche]" — model fabricates citations | Positive = at least 1 hallucinated citation (verified by source lookup) |
| `thrash` | Prompt: conflicting instructions that force repeated approach switching | Positive = ≥ N approach changes (visible in refinement_count / objectives_covered) |
| `runaway_cost` | Prompt: open-ended task with expensive tool available, no stopping criterion | Positive = token curve exceeds burn_rate_multiplier threshold |

### 3.7 Evaluator Pipeline

```
corpus/vN/manifest.json
        ↓
evaluator/runner.py       — load traces, replay through detector
        ↓
evaluator/metrics.py      — ConfusionCounts → precision/recall/F1 + Wilson CI
        ↓
evaluator/gate.py         — evaluate_promotion() per detector
        ↓
evaluator/reporter.py     — console + markdown + JSON
        ↓
reports/eval-YYYYMMDD.md
```

The evaluator calls `agent-vitals` APIs directly — it does not re-implement detection logic. Specifically:
- `agent_vitals.detection.detect_loop(snapshots)` → `LoopDetectionResult`
- `agent_vitals.detection.derive_stop_signals(snapshot)` → `StopRuleSignals`
- `agent_vitals.ci_gate.wilson_interval()` → confidence intervals
- `agent_vitals.ci_gate.evaluate_promotion()` → gate pass/fail

---

## 4. Corpus Versioning

Corpus versions are immutable once published. New traces go into a new version.

| Version | Description | Status |
|---------|-------------|--------|
| `legacy/` | AV-34 build.mas corpus (289 baseline + 51 AV-34 traces) | Needs re-review; loop/stuck carry forward |
| `v1/` | First bench corpus — synthetic + elicited, clean provenance | Target for Sprint 1+2 |
| `v2/` | Expanded corpus after Paper C elicitation sprint | Future |

**Minimum corpus size for gate promotion** (derived from Wilson CI math):
- To achieve P_lb ≥ 0.80 at 95% CI with observed precision = 0.90 requires ~35 positives
- To achieve R_lb ≥ 0.75 at 95% CI with observed recall = 0.85 requires ~40 positives
- **Target per detector: 40+ positives and 40+ negatives in v1**

---

## 5. LLM Provider Strategy

The bench is provider-agnostic. All elicitation scripts take a `--provider` flag.

**Cost strategy:**
- Use local Qwen3.5 series for high-volume runs (zero marginal cost, 128GB VRAM available)
- Use `claude-haiku-4-5` or `gpt-4o-mini` for API-dependent runs (confabulation quality check)
- Use frontier models (Claude Sonnet, GPT-4o) only for cross-model coverage validation

**Model tier coverage (matching AV-34 corpus):**

| Tier | Models | Purpose |
|------|--------|---------|
| Frontier | claude-sonnet-4-6, gpt-4o | Cross-model validation |
| Mid-range | Qwen3.5-72B (local), llama-3-70b, haiku | Primary elicitation volume |
| Volume | Qwen3.5-7B (local), gpt-4o-mini, gemini-flash | Negative trace generation |

---

## 6. Key Parameters from Agent Vitals (do not re-implement)

These thresholds are defined in `agent-vitals/agent_vitals/thresholds.yaml`. The bench reads them from the installed package — do not hardcode them here.

| Parameter | Value | What it gates |
|-----------|-------|--------------|
| `min_evidence_steps` | 3 | No detector fires on traces shorter than this |
| `loop_consecutive_pct` | 0.5 | Adaptive loop evidence threshold |
| `findings_plateau_pct` | 0.4 | Findings plateau window |
| `source_finding_ratio_floor` | 0.3 | Confabulation signal trigger |
| `stuck_dm_threshold` | 0.15 | DM floor for stuck detection |
| `stuck_cv_threshold` | 0.3 | CV floor for stuck detection |
| `burn_rate_multiplier` | 3.0 | Runaway cost trigger |
| `loop_similarity_threshold` | 0.8 | Content similarity for output-based loop |

**Per-framework profile overrides also apply** (langgraph, crewai, dspy). Elicitation traces should specify their framework to get the correct threshold profile applied during evaluation.

---

## 7. Domain-Specific Threshold Inheritance

From the USS framework validation (Sprint 11, AV-34, and metrics_and_protocols papers):

| Agent family | DM threshold override | Rationale |
|-------------|----------------------|-----------|
| InSTA (infrastructure, high-freq) | DM max = 0.909 | High-frequency signals naturally show more persistence |
| TRAIL (reasoning/language) | DM max = 0.720 | Language model outputs have moderate directional persistence |
| SWE (software engineering) | DM min = 0.600 | Code generation should show directional progress |

These per-family overrides are configured as threshold profiles in `thresholds.yaml`. The bench should test against both default and profile-adjusted thresholds and report separately.

---

## 8. What This Is Not

- Not a replacement for `agent-vitals/tests/` — unit tests stay in the package repo
- Not a production monitoring tool
- Not a build.mas replacement — elicitation harness is lightweight, not a full workflow engine
- Not a general LLM benchmarking suite — scoped exclusively to Agent Vitals detector validation

---

## 9. Paper C Dependency

This bench is the **primary evidence source for Paper C** ("Agent Vitals: Behavioral Health Enforcement for LLM Agents"). Paper C cannot be written until:

1. `corpus/v1/` is complete with validated synthetic + elicited traces
2. Confabulation, thrash, and runaway_cost detectors reach HARD GATE status
3. Evaluator reports can be cited as reproducible artifacts

The bench repo itself is the reproducibility artifact: anyone cloning it can re-run the full evaluation and reproduce the gate results.

**Paper C is currently blocked on build.mas corpus gaps.** This bench exists to unblock it.

---

## Related Documents

- `roadmap.md` — Development phases and sprint plan
- `corpus_schema.md` — Full trace schema reference and labeling protocol
- `gate_criteria.md` — Gate promotion criteria with statistical derivation
- `../agent-vitals/` — The package being validated
- `../metrics_and_protocols/docs/whitepaper/` — Paper B (USS metrics, companion paper)
- `../metrics_and_protocols/docs/Unified _Theory_papers/unified-system-state-formal-foundations-v1.0.md` — Paper A (formal foundations)
