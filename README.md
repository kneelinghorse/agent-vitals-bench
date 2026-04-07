# agent-vitals-bench

<!-- Update OWNER/REPO when remote is configured -->
![Gate Evaluation](https://github.com/OWNER/agent-vitals-bench/actions/workflows/gate-eval.yml/badge.svg)

Validation and benchmarking harness for the [agent-vitals](../agent-vitals/) detector package.

**Status:** Active — all 5 detectors pass HARD GATE across 4 framework profiles.

---

## What This Is

A dedicated test apparatus for proving that Agent Vitals detectors work with publication-grade rigor. It exists because the original validation harness (build.mas) was a production LangChain workflow, not a controlled test environment.

This repo is the evidence source for **Paper C: Agent Vitals Empirical Paper**.

## Current Detector Status

Gate thresholds: P_lb >= 0.80, R_lb >= 0.75 (Wilson CI 95%), min 25 positives.

| Detector | P_lb | R_lb | Positives | Gate |
|----------|------|------|-----------|------|
| loop | 0.982 | 0.982 | 212 | PASS |
| stuck | 0.939 | 0.973 | 138 | PASS |
| confabulation | 0.914 | 0.828 | 227 | PASS |
| thrash | 0.978 | 0.978 | 173 | PASS |
| runaway_cost | 0.827 | 0.977 | 163 | PASS |

All 4 framework profiles (default, crewai, langgraph, dspy) pass all enabled gates.

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
