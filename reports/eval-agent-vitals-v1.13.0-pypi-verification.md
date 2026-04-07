# Agent-Vitals v1.13.0 PyPI Post-Release Verification

**Date:** 2026-04-07
**Build:** agent-vitals v1.13.0 (released commit ed055ef = 2fd3a25 + version bump)
**Sprint:** 16, mission s16-m03
**Source:** PyPI wheel `agent_vitals-1.13.0-py3-none-any.whl`
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## TL;DR

- **Default-config bench validation: PASS, bit-identical to 2fd3a25.** All 5 detectors at parity. Composite gate PASS.
- **Bundled TDA model: PASS.** `models/runaway_cost.joblib` is in the wheel at the expected path, loads correctly, and produces the predicted 17 overrides on the bench corpus.
- **2fd3a25 cosmetic fix verified in wheel:** `TDAConfig.min_steps == 7` (not 5).
- **REGRESSION FOUND:** `thresholds.yaml` is **missing from the wheel**. `pyproject.toml:[tool.setuptools.package-data]` only includes `models/*.joblib`, omitting `thresholds.yaml`. Default-config evaluations fall back gracefully (which is why bench's gate passes), but **`VitalsConfig.for_framework("langgraph"|"crewai"|"dspy")` silently returns the default config instead of the framework-specific overrides.** Severity: high for framework profile users, zero for default-config users. Recommend a v1.13.1 patch — one-line `pyproject.toml` fix.

This is exactly the kind of bundling regression s16-m03 was designed to catch.

## Verification Setup

Clean Python 3.12 venv at `/tmp/av-1.13.0-verify/.venv` (separate from both bench main venv and tda-experiment venv). Install command:

```
pip install 'agent-vitals[tda]==1.13.0'
```

Resulting install (relevant packages):

```
agent-vitals-1.13.0
giotto-tda-0.6.2
scikit-learn-1.3.2
joblib-1.5.3
numpy-1.26.4
scipy-1.17.1
giotto-ph-0.2.4
pyflagser-0.4.7
```

No errors during install. All TDA optional extras resolved cleanly.

## Smoke Tests (PASS)

| Check | Expected | Actual |
|-------|----------|--------|
| `agent_vitals.__file__` not from sibling editable install | NOT in `unified-system-state/agent-vitals/` | site-packages path ✓ |
| Installed version | `1.13.0` | `1.13.0` ✓ |
| `models/runaway_cost.joblib` exists in wheel | yes | `194,592` bytes at `…/agent_vitals/models/runaway_cost.joblib` ✓ |
| `TDAConfig.min_steps` (verifies 2fd3a25 cosmetic fix shipped) | `7` | `7` ✓ |
| `is_tda_available()` | `True` (extras installed) | `True` ✓ |
| `predict_runaway_cost` importable | yes | yes ✓ |

## Full-Corpus Validation Against Released Wheel (PASS)

Same validation harness as the iteration chain (`scripts/validate_4855cde.py`), invoked from the clean venv with `PYTHONPATH=<bench>` so bench's `evaluator.runner` and v1 corpus are accessible. The agent-vitals code is the released wheel from PyPI, NOT the editable sibling checkout.

| Detector | TP | FP | FN | TN | P | R | F1 |
|----------|----|----|----|----|----|----|----|
| loop | 212 | 0 | 0 | 1282 | 1.000 | 1.000 | 1.000 |
| stuck | 138 | 4 | 0 | 1352 | 0.972 | 1.000 | 0.986 |
| confabulation | 282 | 4 | 26 | 1182 | 0.986 | 0.916 | **0.9495** |
| thrash | 261 | 0 | 0 | 1233 | 1.000 | 1.000 | 1.000 |
| runaway_cost | 229 | 11 | 0 | 1254 | 0.954 | 1.000 | **0.9765** |

17 TDA overrides applied. **Composite gate: PASS.**

### Bit-identical to 2fd3a25 expected numbers

| Metric | 2fd3a25 (sibling editable) | v1.13.0 PyPI wheel | Δ |
|--------|----------------------------|---------------------|---|
| confab F1 | 0.9495 | 0.9495 | 0 |
| confab TP | 282 | 282 | 0 |
| confab FP | 4 | 4 | 0 |
| runaway_cost F1 | 0.9765 | 0.9765 | 0 |
| runaway_cost TP | 229 | 229 | 0 |
| runaway_cost FP | 11 | 11 | 0 |
| TDA overrides applied | 17 | 17 | 0 |

The released wheel reproduces the validated commit 2fd3a25 numbers exactly. The release pipeline is correct for the default-config evaluation path.

## REGRESSION: `thresholds.yaml` Missing From Wheel

### What I observed

The validation script's first line of output:

```
Thresholds YAML not found at /tmp/av-1.13.0-verify/.venv/lib/python3.12/site-packages/agent_vitals/thresholds.yaml; using defaults
```

Inventory of YAML/JSON/joblib data files in the installed wheel:

```
.../site-packages/agent_vitals/models/runaway_cost.joblib
```

Inventory of YAML files in the agent-vitals source tree:

```
/Users/mac-studio/projects/unified-system-state/agent-vitals/agent_vitals/thresholds.yaml
```

The source tree has `thresholds.yaml`, the wheel does not.

### Root cause

`pyproject.toml`:

```toml
[tool.setuptools.package-data]
agent_vitals = ["models/*.joblib"]
```

`models/*.joblib` is included; `thresholds.yaml` is not. The TDA bundling fix (av-s06-m02) added the `models/*.joblib` glob but didn't sweep up the existing data file.

### Impact

`agent_vitals/config.py` graceful-degrades when `thresholds.yaml` is missing — it logs a "using defaults" message and falls back to the dataclass defaults baked into `VitalsConfig`. This is why bench's default-config validation passes bit-identically.

The silent failure mode is on `VitalsConfig.for_framework("langgraph" | "crewai" | "dspy")`, which is supposed to return a config with framework-specific threshold overrides loaded from `thresholds.yaml`. With the file missing, it returns the default config unchanged.

I confirmed this directly:

```python
default_cfg = VitalsConfig.from_yaml()
# default loop_consecutive_pct: 0.5

for fw in ['langgraph', 'crewai', 'dspy']:
    cfg = default_cfg.for_framework(fw)
    print(f'{fw} loop_consecutive_pct={cfg.loop_consecutive_pct}')

# Output:
# langgraph loop_consecutive_pct=0.5  same_as_default=True
# crewai    loop_consecutive_pct=0.5  same_as_default=True
# dspy      loop_consecutive_pct=0.5  same_as_default=True
```

All three framework profiles silently return default config. No exception, no warning beyond the initial "using defaults" message at module load.

For a user installing `agent-vitals` from PyPI to monitor a langgraph or crewai or dspy agent, this means **the per-framework threshold overrides validated and tuned across Sprints 9-14 are not actually being applied at runtime**. The detector still runs, but with default thresholds rather than the framework-tuned ones. Whether this causes a regression depends on the framework — for DSPy (where Sprint 11's `loop_consecutive_pct` mitigation matters) the impact is probably real.

### The Fix (one line in pyproject.toml)

```diff
 [tool.setuptools.package-data]
-agent_vitals = ["models/*.joblib"]
+agent_vitals = ["models/*.joblib", "thresholds.yaml"]
```

Or, more conservatively to catch any future YAML data files:

```diff
 [tool.setuptools.package-data]
-agent_vitals = ["models/*.joblib"]
+agent_vitals = ["models/*.joblib", "*.yaml"]
```

Then re-cut as **v1.13.1**. No code change required, no re-validation of detector behavior needed — the underlying detector logic is identical to v1.13.0. The only change is that the wheel will now include `thresholds.yaml` and framework profiles will work as designed.

### Recommended verification for v1.13.1

After re-cut, the same s16-m03 verification script will catch the fix:

1. Clean venv → `pip install 'agent-vitals[tda]==1.13.1'`
2. Run `validate_4855cde.py` → expect bit-identical to v1.13.0 numbers (default config path is unchanged)
3. Plus a quick `for_framework("dspy")` check that returns DIFFERENT thresholds from default

The bench validation script does NOT currently exercise the framework-profile path. **Recommended bench-side follow-up:** add a per-framework run to the validation harness so any future regression in `thresholds.yaml` bundling is caught by the gate, not just by manual smoke-test.

## What Worked About v1.13.0

Setting aside the regression: the core release is solid.

- Both M01 (causal confab) and M02 (hybrid runaway TDA) ship as designed
- TDA optional extras install cleanly on Python 3.12
- Bundled `runaway_cost.joblib` is correctly packaged and loadable
- Lazy-loader pattern for TDA backend works in both directions (with extras and without)
- The 2fd3a25 cosmetic `min_steps=7` fix shipped and is observable in `TDAConfig`
- Default-config bench evaluation reproduces 2fd3a25 numbers bit-for-bit
- The v1.13.0 release pipeline (tag → GitHub → PyPI) works correctly for the code paths bench validates

## Recommendation

1. **Cut v1.13.1 as a packaging-only patch.** One-line change to `pyproject.toml`, no code changes, no detector re-validation. Should take less than an hour including release pipeline.
2. **Note the regression in the v1.13.0 release notes / CHANGELOG** so anyone who already installed v1.13.0 knows to upgrade if they're using framework profiles.
3. **Add a framework-profile bench validation step** so the next iteration of s16-m03 (or its v1.13.1 sibling) catches this kind of bundling gap automatically. I'll add it as a Sprint 16 follow-up note.

## Validation Methodology

1. Clean Python 3.12 venv created at `/tmp/av-1.13.0-verify/.venv` (separate from both bench main venv and the tda-experiment venv used for prior validation iterations)
2. `pip install 'agent-vitals[tda]==1.13.0'` from PyPI
3. Smoke tests confirm: NOT editable install, version 1.13.0, bundled `runaway_cost.joblib` present and 194,592 bytes, `is_tda_available()=True`, `TDAConfig.min_steps=7`
4. Full bench corpus validation via `scripts/validate_4855cde.py` with `PYTHONPATH=<bench>` and the clean venv's python
5. Per-detector confusion matrix compared to 2fd3a25 expected numbers — bit-identical
6. Source-read of `pyproject.toml:[tool.setuptools.package-data]` confirms `thresholds.yaml` is NOT in the include list
7. Direct test of `VitalsConfig.for_framework(fw).loop_consecutive_pct` for `langgraph`, `crewai`, `dspy` confirms all three return default value 0.5 (same as `default_cfg.loop_consecutive_pct`)

## Reference Files

- Released wheel: `https://pypi.org/project/agent-vitals/1.13.0/`
- Release commit: `ed055ef` on `https://github.com/kneelinghorse/agent-vitals/`
- Bench validation script: [scripts/validate_4855cde.py](scripts/validate_4855cde.py)
- Iteration chain: [reports/eval-agent-vitals-{8541747,c39ae64,4855cde,2fd3a25}-validation.md](reports/)
- Regression source: `agent_vitals/pyproject.toml:[tool.setuptools.package-data]`
