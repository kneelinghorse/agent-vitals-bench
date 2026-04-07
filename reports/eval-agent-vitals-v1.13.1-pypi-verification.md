# Agent-Vitals v1.13.1 PyPI Post-Release Verification

**Date:** 2026-04-07
**Build:** agent-vitals v1.13.1 (packaging-only patch on top of v1.13.0 / 2fd3a25)
**Sprint:** 16, mission s16-m03 (re-verification)
**Source:** PyPI wheel `agent_vitals-1.13.1-py3-none-any.whl`
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## TL;DR

**v1.13.1 verification: PASS on every check.**

- `thresholds.yaml` is now bundled in the wheel (4,981 bytes at expected path).
- `runaway_cost.joblib` is still bundled (194,592 bytes, unchanged from v1.13.0).
- Default-config bench validation: bit-identical to v1.13.0 (= 2fd3a25). Composite gate PASS.
- **Framework-profile regression resolved.** All three profiles now load their YAML overrides:
  - `langgraph`: 1 field overridden (`loop_consecutive_pct: 0.5 → 0.4`)
  - `crewai`: 2 fields overridden (`burn_rate_multiplier: 2.5 → 3.0`, `token_scale_factor: 1.0 → 0.7`)
  - `dspy`: 3 fields overridden (`loop_consecutive_pct: 0.5 → 0.7`, `stuck_dm_threshold: 0.15 → 0.1`, `workflow_stuck_enabled: research-only → none`)
- Detector logic is bit-identical to v1.13.0 — only `pyproject.toml` and `CHANGELOG.md` were touched, exactly as advertised.

The Sprint 9–14 framework tuning is now actually applied at runtime for users installing from PyPI. v1.13.1 is shippable to framework-profile users.

## Verification Setup

Clean Python 3.12 venv at `/tmp/av-1.13.1-verify/.venv` (separate from bench main venv, tda-experiment venv, and the v1.13.0 verify venv).

```
pip install 'agent-vitals[tda]==1.13.1'
```

Resolved cleanly. Same dependency stack as v1.13.0 verify (giotto-tda 0.6.2, scikit-learn 1.3.2, joblib 1.5.3, numpy 1.26.4, scipy 1.17.1, giotto-ph 0.2.4, pyflagser 0.4.7).

## Smoke Tests (PASS)

| Check | Expected | Actual |
|-------|----------|--------|
| Not from sibling editable install | site-packages path | `/tmp/av-1.13.1-verify/.venv/lib/python3.12/site-packages/agent_vitals/...` ✓ |
| Installed version | `1.13.1` | `1.13.1` ✓ |
| `thresholds.yaml` in wheel | yes | **4,981 bytes** ✓ (was MISSING in v1.13.0) |
| `models/runaway_cost.joblib` in wheel | yes | 194,592 bytes ✓ |
| `is_tda_available()` | `True` | `True` ✓ |
| `TDAConfig.min_steps` (2fd3a25 cosmetic fix) | `7` | `7` ✓ |

## Framework-Profile Re-Verification (PASS)

The v1.13.0 regression diagnostic was that `VitalsConfig.for_framework(...)` silently returned the default config because `thresholds.yaml` was not bundled. Direct re-check on v1.13.1:

| Framework | Fields overridden vs default | Override details |
|---|---|---|
| **langgraph** | 1 | `loop_consecutive_pct: 0.5 → 0.4` |
| **crewai** | 2 | `burn_rate_multiplier: 2.5 → 3.0`, `token_scale_factor: 1.0 → 0.7` |
| **dspy** | 3 | `loop_consecutive_pct: 0.5 → 0.7`, `stuck_dm_threshold: 0.15 → 0.1`, `workflow_stuck_enabled: research-only → none` |

Each profile now diverges from the default in the way its `thresholds.yaml` entry specifies. Sprint 9–14 framework tuning is restored at runtime.

(The crewai entry happens not to override `loop_consecutive_pct` — the divergence is on `burn_rate_multiplier` and `token_scale_factor` instead. The v1.13.0 verification only checked `loop_consecutive_pct`, so the fact that v1.13.1 crewai diverges on a *different* field is an even stronger signal that the YAML is being read in full, not just one key.)

## Full-Corpus Validation Against Released Wheel (PASS)

Same harness as the iteration chain (`scripts/validate_4855cde.py`), invoked from the clean v1.13.1 venv with `PYTHONPATH=<bench>` so bench's `evaluator.runner` and v1 corpus are accessible. The agent-vitals code is the released wheel from PyPI.

| Detector | TP | FP | FN | TN | P | R | F1 |
|----------|----|----|----|----|----|----|----|
| loop | 212 | 0 | 0 | 1282 | 1.000 | 1.000 | 1.000 |
| stuck | 138 | 4 | 0 | 1352 | 0.972 | 1.000 | 0.986 |
| confabulation | 282 | 4 | 26 | 1182 | 0.986 | 0.916 | **0.9495** |
| thrash | 261 | 0 | 0 | 1233 | 1.000 | 1.000 | 1.000 |
| runaway_cost | 229 | 11 | 0 | 1254 | 0.954 | 1.000 | **0.9765** |

17 TDA overrides applied. **Composite gate: PASS.**

### Bit-identical to v1.13.0 (and to 2fd3a25)

| Metric | v1.13.0 PyPI | v1.13.1 PyPI | Δ |
|--------|--------------|--------------|---|
| confab F1 | 0.9495 | 0.9495 | 0 |
| confab TP | 282 | 282 | 0 |
| confab FP | 4 | 4 | 0 |
| runaway_cost F1 | 0.9765 | 0.9765 | 0 |
| runaway_cost TP | 229 | 229 | 0 |
| runaway_cost FP | 11 | 11 | 0 |
| TDA overrides | 17 | 17 | 0 |

Detector logic unchanged, exactly as advertised. The packaging fix has zero impact on detector behavior — it only restores the framework-profile code path.

## What v1.13.1 Restores

For users installing `agent-vitals` from PyPI to monitor a langgraph / crewai / dspy agent, the per-framework threshold overrides validated and tuned across Sprints 9–14 are now actually being applied at runtime. Specifically:

- **dspy**: Sprint 11's `loop_consecutive_pct` mitigation (0.5 → 0.7) is back in effect, plus the dspy-specific stuck detector tuning.
- **langgraph**: tightened `loop_consecutive_pct` (0.5 → 0.4) is back in effect.
- **crewai**: lenient `burn_rate_multiplier` (2.5 → 3.0) and reduced `token_scale_factor` (1.0 → 0.7) are back in effect.

All three were silently no-ops in v1.13.0.

## Bench-Side Follow-Up Status

The Sprint 16 follow-up to add a per-framework run to the bench validation harness is still on the backlog — the manual smoke test in this report (the field-divergence enumerator) is the prototype for what the harness gate should automate.

## Validation Methodology

1. Clean Python 3.12 venv at `/tmp/av-1.13.1-verify/.venv` (separate from bench, tda-experiment, and v1.13.0 verify venvs).
2. `pip install 'agent-vitals[tda]==1.13.1'` from PyPI — no errors.
3. Smoke tests: NOT editable install ✓, version 1.13.1 ✓, `thresholds.yaml` present at 4,981 bytes ✓, `runaway_cost.joblib` present at 194,592 bytes ✓, `is_tda_available()=True` ✓, `TDAConfig.min_steps=7` ✓.
4. Framework-profile divergence enumeration: for each of langgraph/crewai/dspy, walk every `VitalsConfig` field and report any value that differs from the default. All three profiles diverge on at least one field.
5. Full bench corpus validation via `scripts/validate_4855cde.py` with `PYTHONPATH=<bench>` and the clean v1.13.1 venv's python.
6. Per-detector confusion matrix compared to v1.13.0 / 2fd3a25 expected numbers — bit-identical.

## Recommendation

**Ship v1.13.1.** Bundling regression is closed, detector logic is unchanged, framework profiles work as designed for the first time since the v1.13.0 release. No further iterations needed on this thread from the bench side.

## Reference Files

- Released wheel: `https://pypi.org/project/agent-vitals/1.13.1/`
- v1.13.0 verification report (regression diagnosis): [reports/eval-agent-vitals-v1.13.0-pypi-verification.md](eval-agent-vitals-v1.13.0-pypi-verification.md)
- Bench validation script: [scripts/validate_4855cde.py](../scripts/validate_4855cde.py)
- Iteration chain: [reports/eval-agent-vitals-{8541747,c39ae64,4855cde,2fd3a25}-validation.md](.)
