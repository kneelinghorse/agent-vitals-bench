# Cross-Framework Gate Report — Post-Replay-Audit

**Generated:** 2026-04-08T03:58:57Z
**Corpus:** v1, min_confidence ≥ 0.80
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost (canonical bench five)
**Profiles:** default, crewai, dspy, langgraph (default = no framework override; the other three apply the agent-vitals thresholds.yaml profile via the v1.14.0 public VitalsConfig profile-introspection API)
**Runtime modes:** default (`tda_enabled=False`, pure handcrafted rules) and tda (`tda_enabled=True`, handcrafted with TDA persistence-feature gradient-boosting override on runaway_cost only)
**Replay backend:** canonical `agent_vitals.backtest._replay_trace` via bench's repaired `evaluator/runner.replay_trace`. No mirror, no drift, no per-script ad-hoc replay paths.

## TL;DR — Composite gate grid

Counts are non-excluded detectors only. DSPy intentionally disables `workflow_stuck_enabled`, so its `stuck` detector is excluded from the composite gate (documented framework-specific integration constraint, not a detector failure).

| Profile | default mode | tda mode |
|---|---|---|
| default | 4/5 NO-GO | 5/5 PASS |
| crewai | 4/5 NO-GO | 5/5 PASS |
| dspy | 3/4 NO-GO | 4/4 PASS |
| langgraph | 4/5 NO-GO | 5/5 PASS |

## TL;DR — runaway_cost precision lower bound

Cross-cut on the load-bearing number for the m02 §8.4 / m03 §4.3 discussion. Publication HARD GATE threshold is P_lb ≥ 0.80.

| Profile | default mode P_lb | tda mode P_lb |
|---|---|---|
| default | 0.7654 | 0.9198 |
| crewai | 0.7443 | 0.9111 |
| dspy | 0.7926 | 0.8760 |
| langgraph | 0.7443 | 0.9111 |

## Methodology

Each (profile, runtime_mode) cell is one full pass through `evaluator.runner.run_evaluation` on corpus v1 with `min_confidence=0.80`. Profile selection threads through `VitalsConfig.for_framework(profile)` exactly as production does. Runtime mode is enforced explicitly: `default` forces `tda_enabled=False`, `tda` forces `tda_enabled=True`. The trace loop and confusion-matrix accounting are identical across all 8 cells — only the config differs.

Manifest filtering also threads the profile: when a profile is active, traces tagged with a *different* framework are excluded, but untagged traces are kept. This means the per-profile trace counts may differ slightly from the default-profile baseline; the trace_count line in each section reports the actual count.

**Replay backend.** Both runtime modes go through `agent_vitals.backtest._replay_trace` via bench's repaired `evaluator/runner.replay_trace`. This is the same canonical production replay path that the v1.14.0 acceptance and the regenerated five-paradigm comparative report both rely on. The pre-audit per-framework reports under `reports/2026-04-07T14:45*` predate the replay repair and should be considered superseded by this report for any paper claim.

**Gate thresholds.** From `evaluator/gate.py`: `MIN_POSITIVES=25`, `MIN_PRECISION_LB=0.80`, `MIN_RECALL_LB=0.75` (Wilson 95% lower bounds, two-sided). A detector passes if it has enough positives *and* both lower bounds clear thresholds. The composite gate is the AND of all non-excluded detectors.

## Per-profile detail

## Profile: default

- **default runtime mode** — composite 4/5 (NO-GO), trace_count=1494
- **tda runtime mode** — composite 5/5 (PASS), trace_count=1494

### default × default runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 212 | 0 | 0 | 1282 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.9822 |
| stuck | HARD GATE | 138 | 4 | 0 | 1352 | 0.9718 | 1.0000 | 0.9857 | 0.9298 | 0.9729 |
| confabulation | HARD GATE | 282 | 4 | 26 | 1182 | 0.9860 | 0.9156 | 0.9495 | 0.9646 | 0.8792 |
| thrash | HARD GATE | 261 | 0 | 0 | 1233 | 1.0000 | 1.0000 | 1.0000 | 0.9855 | 0.9855 |
| runaway_cost | NO-GO | 229 | 52 | 0 | 1213 | 0.8149 | 1.0000 | 0.8980 | 0.7654 | 0.9835 |

### default × tda runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 212 | 0 | 0 | 1282 | 1.0000 | 1.0000 | 1.0000 | 0.9822 | 0.9822 |
| stuck | HARD GATE | 138 | 4 | 0 | 1352 | 0.9718 | 1.0000 | 0.9857 | 0.9298 | 0.9729 |
| confabulation | HARD GATE | 282 | 4 | 26 | 1182 | 0.9860 | 0.9156 | 0.9495 | 0.9646 | 0.8792 |
| thrash | HARD GATE | 261 | 0 | 0 | 1233 | 1.0000 | 1.0000 | 1.0000 | 0.9855 | 0.9855 |
| runaway_cost | HARD GATE | 229 | 11 | 0 | 1254 | 0.9542 | 1.0000 | 0.9765 | 0.9198 | 0.9835 |

## Profile: crewai

- **default runtime mode** — composite 4/5 (NO-GO), trace_count=1334
- **tda runtime mode** — composite 5/5 (PASS), trace_count=1334

### crewai × default runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 184 | 0 | 0 | 1150 | 1.0000 | 1.0000 | 1.0000 | 0.9795 | 0.9795 |
| stuck | HARD GATE | 120 | 1 | 0 | 1213 | 0.9917 | 1.0000 | 0.9959 | 0.9547 | 0.9690 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1058 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1097 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | NO-GO | 205 | 52 | 0 | 1077 | 0.7977 | 1.0000 | 0.8874 | 0.7443 | 0.9816 |

### crewai × tda runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 184 | 0 | 0 | 1150 | 1.0000 | 1.0000 | 1.0000 | 0.9795 | 0.9795 |
| stuck | HARD GATE | 120 | 1 | 0 | 1213 | 0.9917 | 1.0000 | 0.9959 | 0.9547 | 0.9690 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1058 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1097 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | HARD GATE | 205 | 11 | 0 | 1118 | 0.9491 | 1.0000 | 0.9739 | 0.9111 | 0.9816 |

## Profile: dspy

- **default runtime mode** — composite 3/4 (NO-GO), trace_count=1312 — excluded: stuck
- **tda runtime mode** — composite 4/4 (PASS), trace_count=1312 — excluded: stuck

### dspy × default runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 180 | 0 | 0 | 1132 | 1.0000 | 1.0000 | 1.0000 | 0.9791 | 0.9791 |
| stuck | EXCLUDED | 0 | 0 | 102 | 1210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1036 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1075 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | NO-GO | 205 | 38 | 0 | 1069 | 0.8436 | 1.0000 | 0.9152 | 0.7926 | 0.9816 |

*Excluded from composite gate (disabled by profile): stuck.*

### dspy × tda runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 180 | 0 | 0 | 1132 | 1.0000 | 1.0000 | 1.0000 | 0.9791 | 0.9791 |
| stuck | EXCLUDED | 0 | 0 | 102 | 1210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1036 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1075 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | HARD GATE | 205 | 18 | 0 | 1089 | 0.9193 | 1.0000 | 0.9579 | 0.8760 | 0.9816 |

*Excluded from composite gate (disabled by profile): stuck.*

## Profile: langgraph

- **default runtime mode** — composite 4/5 (NO-GO), trace_count=1334
- **tda runtime mode** — composite 5/5 (PASS), trace_count=1334

### langgraph × default runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 184 | 0 | 0 | 1150 | 1.0000 | 1.0000 | 1.0000 | 0.9795 | 0.9795 |
| stuck | HARD GATE | 120 | 0 | 0 | 1214 | 1.0000 | 1.0000 | 1.0000 | 0.9690 | 0.9690 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1058 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1097 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | NO-GO | 205 | 52 | 0 | 1077 | 0.7977 | 1.0000 | 0.8874 | 0.7443 | 0.9816 |

### langgraph × tda runtime mode

| Detector | Status | TP | FP | FN | TN | P | R | F1 | P_lb | R_lb |
|---|---|---|---|---|---|---|---|---|---|---|
| loop | HARD GATE | 184 | 0 | 0 | 1150 | 1.0000 | 1.0000 | 1.0000 | 0.9795 | 0.9795 |
| stuck | HARD GATE | 120 | 0 | 0 | 1214 | 1.0000 | 1.0000 | 1.0000 | 0.9690 | 0.9690 |
| confabulation | HARD GATE | 246 | 4 | 26 | 1058 | 0.9840 | 0.9044 | 0.9425 | 0.9596 | 0.8636 |
| thrash | HARD GATE | 237 | 0 | 0 | 1097 | 1.0000 | 1.0000 | 1.0000 | 0.9840 | 0.9840 |
| runaway_cost | HARD GATE | 205 | 11 | 0 | 1118 | 0.9491 | 1.0000 | 0.9739 | 0.9111 | 0.9816 |

## Reproducibility

```
PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python scripts/regen_cross_framework.py
```

Both modes require the TDA backend (the default-mode cells don't use it, but the script verifies availability at startup so all 8 cells run from the same interpreter and the report is internally consistent). Use the tda-experiment Python 3.12 venv. The machine-readable cell payload is at `reports/eval-cross-framework-v1.json` — every figure in this report is regenerable from it without re-running the detectors.

