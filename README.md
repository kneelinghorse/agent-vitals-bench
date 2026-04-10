# agent-vitals-bench

![Gate Evaluation](https://github.com/kneelinghorse/agent-vitals-bench/actions/workflows/gate-eval.yml/badge.svg)

Validation and benchmarking harness for the [agent-vitals](../agent-vitals/) detector package.

**Status:** Active — composite gate PASS on all profiles in TDA-enhanced mode; default mode has a known runaway_cost precision gap (see below).

---

## What This Is

A dedicated test apparatus for proving that Agent Vitals detectors work with publication-grade rigor. It exists because the original validation harness (build.mas) was a production LangChain workflow, not a controlled test environment.

This repo is the evidence source for **Paper C: Agent Vitals Empirical Paper**.

## Current Detector Status

Gate thresholds: P_lb >= 0.80, R_lb >= 0.75 (Wilson CI 95%), min 25 positives.

All numbers below are from the post-replay-audit cross-framework evaluation on corpus v1 (`min_confidence >= 0.80`). Two runtime modes are evaluated:

- **default** (`tda_enabled=False`): pure handcrafted detection rules — the out-of-box product default.
- **tda** (`tda_enabled=True`): handcrafted rules with TDA persistence-feature gradient-boosting override on runaway_cost.

Source: `reports/eval-cross-framework-v1.{md,json}`

### Default profile, default runtime mode — composite FAIL

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.982 | 0.982 | 212 | PASS |
| stuck | 0.930 | 0.973 | 138 | PASS |
| confabulation | 0.965 | 0.879 | 308 | PASS |
| thrash | 0.986 | 0.986 | 261 | PASS |
| runaway_cost | 0.765 | 0.984 | 229 | **NO-GO** |

### Default profile, TDA runtime mode — composite PASS

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.982 | 0.982 | 212 | PASS |
| stuck | 0.930 | 0.973 | 138 | PASS |
| confabulation | 0.965 | 0.879 | 308 | PASS |
| thrash | 0.986 | 0.986 | 261 | PASS |
| runaway_cost | 0.920 | 0.984 | 229 | PASS |

### Per-framework profile composite

| Profile | Mode | loop | stuck | confab | thrash | runaway | Composite |
|---------|------|------|-------|--------|--------|---------|-----------|
| default | default | PASS | PASS | PASS | PASS | **NO-GO** (P_lb=0.765) | **FAIL** |
| default | tda | PASS | PASS | PASS | PASS | PASS | **PASS** |
| langgraph | default | PASS | PASS | PASS | PASS | **NO-GO** (P_lb=0.744) | **FAIL** |
| langgraph | tda | PASS | PASS | PASS | PASS | PASS | **PASS** |
| crewai | default | PASS | PASS | PASS | PASS | **NO-GO** (P_lb=0.744) | **FAIL** |
| crewai | tda | PASS | PASS | PASS | PASS | PASS | **PASS** |
| dspy | default | PASS | EXCLUDED | PASS | PASS | **NO-GO** (P_lb=0.793) | **FAIL** |
| dspy | tda | PASS | EXCLUDED | PASS | PASS | PASS | **PASS** |

**Key finding:** In default mode, runaway_cost fails the precision gate on every profile (P_lb range 0.744–0.793 vs threshold 0.80). The handcrafted runaway_cost path produces 38–52 false positives per profile. With TDA enabled, false positives drop to 11–18 and all profiles clear the gate.

- **dspy stuck EXCLUDED**: dspy disables stuck entirely (`workflow_stuck_enabled=none`), so stuck is excluded from the composite gate — this is an intentional framework-specific integration constraint, not a detector failure.
- **Upstream coordination**: s18-m04 tracks the request for agent-vitals to harden the default-mode runaway_cost path so it can reach HARD GATE without requiring TDA, or to explicitly re-bless TDA as a required production component.

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
