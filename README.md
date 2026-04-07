# agent-vitals-bench

![Gate Evaluation](https://github.com/kneelinghorse/agent-vitals-bench/actions/workflows/gate-eval.yml/badge.svg)

Validation and benchmarking harness for the [agent-vitals](../agent-vitals/) detector package.

**Status:** Active — composite gate PASS on the default profile; two known per-profile gaps under investigation (see below).

---

## What This Is

A dedicated test apparatus for proving that Agent Vitals detectors work with publication-grade rigor. It exists because the original validation harness (build.mas) was a production LangChain workflow, not a controlled test environment.

This repo is the evidence source for **Paper C: Agent Vitals Empirical Paper**.

## Current Detector Status

Gate thresholds: P_lb >= 0.80, R_lb >= 0.75 (Wilson CI 95%), min 25 positives.

### Default profile — composite PASS

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.982 | 0.982 | 212 | PASS |
| stuck | 0.930 | 0.973 | 138 | PASS |
| confabulation | 0.965 | 0.879 | 308 | PASS |
| thrash | 0.986 | 0.986 | 261 | PASS |
| runaway_cost | 0.847 | 0.984 | 229 | PASS |

### Per-framework profile composite

| Profile | loop | stuck | confab | thrash | runaway | Composite |
|---------|------|-------|--------|--------|---------|-----------|
| default | PASS | PASS | PASS | PASS | PASS | **PASS** |
| langgraph | PASS | PASS | PASS | PASS | PASS | **PASS** |
| crewai | PASS | **NO-GO** (P_lb=0.702) | PASS | PASS | PASS | **FAIL** |
| dspy | PASS | EXCLUDED | PASS | PASS | **NO-GO** (P_lb=0.793) | **FAIL** |

Two known per-profile gaps, both pre-existing in the corpus prior to the v1.13.x release validation cycle:

- **crewai stuck**: 35 false positives over 120 true positives (P_lb=0.702 vs target 0.80). crewai's profile only overrides `burn_rate_multiplier` and `token_scale_factor` — it does not touch any stuck thresholds — so the FP rate is structurally driven by the corpus subset rather than threshold drift. ~5 fewer FPs would clear the gate.
- **dspy runaway_cost**: 38 false positives over 205 true positives (P_lb=0.793 vs target 0.80). dspy disables stuck entirely (`workflow_stuck_enabled=none`), which removes the stuck-suppresses-runaway co-occurrence path in evaluator/runner.py. Every burn_rate_anomaly that would normally be suppressed by a co-firing stuck becomes a runaway FP under dspy. ~6 fewer FPs would clear the gate.

Both issues are tracked for investigation; the v1.13.x production stack is solid for default-config users (the dominant deployment) and for langgraph users.

## Repository Layout

```
agent-vitals-bench/
├── docs/
│   ├── technical_architecture.md   # System design, component relationships
│   ├── roadmap.md                  # Development phases and exit criteria
│   ├── corpus_schema.md            # Trace format, labeling protocol
│   └── gate_criteria.md            # Wilson CI gate math and baselines
├── generators/                     # Synthetic trace factories (Phase 0-1)
├── corpus/                         # Versioned labeled trace corpus (Phase 0+)
├── evaluator/                      # Runner, metrics, gate, reporter (Phase 0)
├── elicitation/                    # LLM elicitation scripts (Phase 2)
└── reports/                        # Generated evaluation reports
```

## Sibling Repos

```
metrics-protocols/
├── agent-vitals/          # The pip package (pip install agent-vitals)
├── agent-vitals-bench/    # This repo
├── metrics_and_protocols/ # Research, papers, CMOS project management
└── build.mas/             # Original LangChain harness (legacy, not used here)
```

## Getting Started

Dependencies: `agent-vitals >= 1.11.0`, `python >= 3.10`

```bash
pip install -e "../agent-vitals[dev]"
pip install -e ".[dev]"
make evaluate                                      # Full gate evaluation (all detectors, default profile)
make evaluate PROFILE=dspy                         # Single profile
make evaluate CORPUS=v1 DETECTORS=loop,stuck       # Targeted run
make test                                          # pytest tests/ -v
make lint                                          # ruff check + format check
```

## CI/CD

The [Gate Evaluation](.github/workflows/gate-eval.yml) workflow runs on every push and PR to `main`. It evaluates all 4 framework profiles in a matrix strategy and fails if any gate drops below threshold. Gate reports are uploaded as artifacts.

## Key Documents

- [Technical Architecture](docs/technical_architecture.md)
- [Roadmap](docs/roadmap.md)
- [Corpus Schema and Labeling Protocol](docs/corpus_schema.md)
- [Gate Criteria](docs/gate_criteria.md)
