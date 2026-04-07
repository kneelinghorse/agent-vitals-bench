# Agent Vitals Bench — Agent Configuration

**Project Name**: Agent Vitals Bench
**Project Type**: Python validation/benchmarking harness
**Primary Language**: Python 3.10+
**Dependencies**: agent-vitals >= 1.11.0, pytest, ruff, mypy

**Description**: Publication-grade empirical validation harness for the agent-vitals detector package. Produces Wilson CI gate reports as primary evidence for Paper C (Agent Vitals Empirical Paper). Tests 5 detectors — loop, stuck, confabulation, thrash, runaway_cost — against labeled trace corpora using synthetic generation, real LLM elicitation, and legacy migration.

---

## Ecosystem Context

This repo is one of three sibling projects under `unified-system-state/`:

```
unified-system-state/
├── metrics_and_protocols/   # Research origin — protocol theory, metrics catalog,
│                            #   whitepapers, telemetry probe, CMOS project mgmt
├── agent-vitals/            # Production pip package (pip install agent-vitals)
│                            #   5 detectors, framework integrations, CI gate math
└── agent-vitals-bench/      # THIS REPO — empirical validation harness
```

### How We Relate

- **metrics_and_protocols** is the research mothership. It holds the protocol theory, metrics catalog, and whitepaper drafts. It defines *what* to measure and *why*.
- **agent-vitals** is the distilled production package. It implements the detectors, thresholds, and `wilson_interval` CI gate math. It defines *how* to detect.
- **agent-vitals-bench** (this repo) is the independent test apparatus. It proves the detectors work with publication-grade rigor. It defines *whether* detection actually works.

### Cross-Project Feedback Loop

All three projects feed each other:

1. **Bench findings flow upstream**: When bench evaluation reveals detector weaknesses (e.g., confabulation R_lb tight at 0.807), that feeds back into `agent-vitals` for detector tuning and into `metrics_and_protocols` for protocol refinement.
2. **Threshold changes flow downstream**: When `agent-vitals` updates `thresholds.yaml` or detector logic, bench must re-evaluate to confirm gates still hold.
3. **Protocol insights flow laterally**: When `metrics_and_protocols` refines elicitation protocols or metric definitions, bench incorporates them into generators and evaluator logic.

When making decisions here, consider impact on all three repos. Surface cross-project implications in mission notes and decisions.

---

## Build & Development Commands

### Installation
```bash
# Install agent-vitals (sibling) + this package in dev mode
make install
# Or manually:
pip install -e "../agent-vitals[dev]"
pip install -e ".[dev]"
```

### Core Commands
```bash
make evaluate                              # Full gate evaluation (corpus v1, all detectors)
make evaluate CORPUS=legacy DETECTORS=loop,stuck   # Targeted run
make test                                  # pytest tests/ -v --tb=short
make lint                                  # ruff check + format check
make typecheck                             # mypy strict mode
make clean                                 # Remove caches and build artifacts
```

### Elicitation (optional deps)
```bash
pip install -e ".[elicitation]"            # openai, anthropic, httpx
python scripts/run_elicitation.py          # Local llama.cpp elicitation
python scripts/batch_elicit_multi.py       # Multi-provider API elicitation
```

---

## Project Structure

```
agent-vitals-bench/
├── evaluator/              # Runner, metrics, gate checks, report generation
│   ├── runner.py           # CLI entry: --corpus, --detectors, --min-confidence
│   ├── metrics.py          # DetectorMetrics, Wilson CI, precision/recall/F1
│   ├── gate.py             # HARD GATE checks (P_lb>=0.80, R_lb>=0.75, min 25 pos)
│   └── reporter.py         # Markdown + JSON report output
├── generators/             # Synthetic trace factories (one per detector)
│   ├── base.py             # TraceGenerator ABC, TraceMetadata dataclass
│   ├── loop.py
│   ├── stuck.py
│   ├── confabulation.py
│   ├── thrash.py
│   └── runaway_cost.py
├── elicitation/            # Real LLM trace elicitation
│   ├── providers.py        # 6 providers: LocalLlamaCpp, OpenAI, Anthropic, OpenRouter, MiniMax, DeepSeek
│   ├── elicit_confabulation.py
│   ├── elicit_thrash.py
│   └── elicit_runaway_cost.py
├── corpus/                 # Versioned labeled trace corpora
│   ├── v1/                 # Primary gate corpus (synthetic + elicited + migrated)
│   │   ├── traces/{detector}/{positive,negative}/
│   │   ├── manifest.json
│   │   └── elicited_*.json
│   └── legacy/             # AV-34 re-reviewed traces (loop, stuck only)
├── scripts/                # Batch generation, elicitation, migration utilities
├── reports/                # Generated gate reports (eval-YYYYMMDD-{corpus}.{md,json})
├── tests/                  # pytest suite
├── docs/                   # Architecture, roadmap, corpus schema, gate criteria
├── agents.md               # THIS FILE
└── cmos/                   # CMOS project management (DO NOT write application code here)
```

---

## Coding Standards

### Python Style
- **Formatter/Linter**: ruff (line-length 100, target py310)
- **Type checking**: mypy strict mode
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: stdlib, third-party, local — separated by blank lines
- No wildcard imports. No unused imports.

### Key Patterns
- **Wilson CI lower bounds** are the publication metric — never use raw P/R for gate decisions
- **Confidence tiers**: 1.0 = synthetic, 0.9 = clean elicited, 0.8 = ambiguous elicited, 0.7 = legacy
- **Evaluator filters at min_confidence >= 0.8** by default
- **Co-occurrence resolution**: mirrors agent-vitals `backtest.py` exactly — stuck suppression, burn_rate-to-runaway bridging, loop+stuck tiebreaker
- **source_productive gate**: runaway/confab generators must maintain coverage>=0.5, sources>=10, findings>=5 during failure phase to avoid stuck cross-triggers

### Testing
- **Framework**: pytest >= 8.0, pytest-cov >= 4.1
- **Location**: `tests/` (never in `cmos/`)
- **Naming**: `test_*.py`, functions `test_*`
- **Run**: `make test`
- All detectors that pass HARD GATE must have regression tests
- New evaluator features need unit tests at minimum
- Elicitation modules: test the logic, mock the LLM calls

### Quality Gates (before completing any mission)
- `make test` passes
- `make lint` passes
- `make typecheck` passes (or known exceptions documented)
- Gate reports regenerated if corpus or evaluator changed

---

## Detector Gate Status

HARD GATE thresholds: **P_lb >= 0.80, R_lb >= 0.75** (Wilson CI 95%), min 25 positives per detector.

| Detector | Gate Status | Notes |
|---|---|---|
| loop | HARD GATE | Passing. Regression guard only. |
| stuck | HARD GATE | Passing. Regression guard only. |
| confabulation | HARD GATE | Tightest margin — R_lb=0.807 (20 FN / 153 pos) |
| thrash | HARD GATE | Reached via elicited corpus |
| runaway_cost | HARD GATE | Reached via elicited corpus |

---

## Key Decisions

- **NO-GO failures were corpus gaps, not detector bugs** — fix by targeted elicitation, not detector changes
- **Zero-cost-first elicitation**: local Qwen3.5 via llama.cpp primary; API providers for cross-model validation only
- **Three-source corpus**: synthetic (generators) + elicited (real LLM) + legacy (AV-34 migration) merged into unified v1
- **Confabulation is model-aware**: different models confabulate differently. Protocols must account for this.

---

## Notes for AI Agents

### Before You Code
1. Read this file (`agents.md`) for project context and standards
2. Check `cmos_agent_onboard()` for current sprint, pending missions, blockers
3. Understand the cross-project ecosystem — changes here may affect `agent-vitals` or `metrics_and_protocols`

### Critical Rules
- Never write application code in `cmos/` — that directory is for project management only
- Never hardcode thresholds — they come from `agent-vitals` `thresholds.yaml`
- Always use Wilson CI lower bounds for gate decisions, never raw precision/recall
- Run `make test && make lint` before completing any mission
- When evaluation results change, regenerate gate reports via `make evaluate`

### Cross-Project Awareness
- If you discover a detector issue, note it for upstream feedback to `agent-vitals`
- If you find a protocol gap, note it for `metrics_and_protocols`
- Reference sibling repos at `../agent-vitals/` and `../metrics_and_protocols/`
- The `agent-vitals` package is installed from source (`pip install -e "../agent-vitals[dev]"`) — changes there are immediately visible here
