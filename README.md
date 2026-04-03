# agent-vitals-bench

Validation and benchmarking harness for the [agent-vitals](../agent-vitals/) detector package.

**Status:** Pre-build — foundational docs complete, awaiting CMOS initialization and Sprint 0.

---

## What This Is

A dedicated test apparatus for proving that Agent Vitals detectors work with publication-grade rigor. It exists because the original validation harness (build.mas) was a production LangChain workflow, not a controlled test environment.

This repo is the evidence source for **Paper C: Agent Vitals Empirical Paper**.

## Current Detector Status

| Detector | AV-34 Status | Bench Target |
|----------|-------------|--------------|
| loop | HARD GATE ✓ | Reproduce + regression guard |
| stuck | HARD GATE ✓ | Reproduce + regression guard |
| confabulation | NO-GO (P_lb=0.790) | Reach HARD GATE with elicited corpus |
| thrash | NO-GO (0 positives) | Reach HARD GATE with elicited corpus |
| runaway_cost | NO-GO (R_lb=0.387) | Reach HARD GATE with elicited corpus |

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

*Build instructions will be added when the repo is initialized (Phase 0, Sprint 0).*

Dependencies: `agent-vitals >= 1.11.0`, `python >= 3.10`

```bash
pip install -e "../agent-vitals[dev]"
pip install -e ".[dev]"
make evaluate CORPUS=legacy DETECTORS=loop,stuck  # calibration check
```

## Key Documents

- [Technical Architecture](docs/technical_architecture.md)
- [Roadmap](docs/roadmap.md)
- [Corpus Schema and Labeling Protocol](docs/corpus_schema.md)
- [Gate Criteria](docs/gate_criteria.md)
