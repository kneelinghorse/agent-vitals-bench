# Agent-Vitals c39ae64 Validation Report

**Date:** 2026-04-07
**Build:** agent-vitals commit c39ae64 (vestigial latest-window gate removal)
**Sprint:** 15 (sixth bench validation iteration)
**Corpus:** v1, 1494 traces, min_confidence >= 0.8

## TL;DR — PASS, exact bench parity

c39ae64 lands at TP=282, FP=4, F1=0.9495, exactly matching the bench prototype reference and the prediction from the 8541747 validation report. Composite HARD GATE PASS. All other detectors at parity. **This is the release candidate.**

## Gate Results

| Detector | P | R | F1 | P_lb | R_lb | Positives | Status |
|----------|---|---|----|------|------|-----------|--------|
| loop | 1.000 | 1.000 | 1.000 | 0.982 | 0.982 | 212 | HARD GATE |
| stuck | 0.972 | 1.000 | 0.986 | 0.930 | 0.973 | 138 | HARD GATE |
| confabulation | 0.986 | 0.916 | **0.950** | 0.965 | 0.879 | 308 | **HARD GATE** |
| thrash | 1.000 | 1.000 | 1.000 | 0.986 | 0.986 | 261 | HARD GATE |
| runaway_cost | 0.891 | 1.000 | 0.942 | 0.847 | 0.984 | 229 | HARD GATE |

**Composite gate: PASS**

## Confab Detail — Full Iteration Chain

| Metric | b3436ef | f6c73f8 | 78d2896 | 55dbb79 | 8541747 | **c39ae64** | Bench reference |
|--------|---------|---------|---------|---------|---------|-------------|-----------------|
| TP | — | 288 | 288 | 288 | 211 | **282** | 282 |
| FP | — | 15 | 14 | 14 | 3 | **4** | 4 |
| FN | — | 20 | 20 | 20 | 97 | **26** | 26 |
| F1 | 0.860 | 0.943 | 0.944 | 0.944 | 0.808 | **0.9495** | 0.9495 |
| Precision | 0.812 | 0.950 | 0.954 | 0.954 | 0.986 | **0.986** | 0.986 |
| Recall | — | 0.935 | 0.935 | 0.935 | 0.685 | **0.9156** | 0.9156 |

c39ae64 achieves **bit-identical numbers** to the bench prototype reference. The prediction from the 8541747 report (TP=282, FP=4, F1=0.9495) was on the nose.

## Targets vs Actuals

| Target | Required | Actual | Status |
|--------|----------|--------|--------|
| Confab F1 | ≥ 0.949 | 0.9495 | PASS |
| Confab FP | ≤ 5 | 4 | PASS |
| Confab P_lb | ≥ 0.8 | 0.965 | PASS (16.5pp buffer) |
| Confab R_lb | ≥ 0.75 | 0.879 | PASS (12.9pp buffer) |
| All other detectors at parity | — | All identical to 55dbb79 | PASS |

## Architectural Outcome

The pair (8541747 final-step adjudication + c39ae64 latest-window gate removal) is the right factoring of the streaming-vs-one-shot reconciliation:

- **`_replay_trace`** owns the trace-label semantics. It calls `detect_loop` per snapshot for streaming early-warning, then applies a final-step override at the trace boundary so the trace label equals the bench reference one-shot.
- **`_detect_causal_confabulation`** owns the per-call detection logic. It is now free of streaming-mode workarounds — the same code path produces the right answer whether called once with the full trace or per-step in replay mode.
- **Per-snapshot `confabulation_detected` flags** continue to flow as before for early-warning consumers (monitor.py, intervention paths). The "minimum-blast-radius" property holds: only the trace label changed.

This is cleaner than either single change in isolation. 78d2896's per-step gate was a workaround for a problem that needed to be solved at the trace-label layer (which 8541747 now does). With both layers in their right places, the per-call detector and the trace aggregator are both simpler and bench-equivalent.

## Validation Methodology

1. agent-vitals checked out at HEAD = c39ae64.
2. Bench `evaluator/runner.py` `replay_trace` mirror retained the 8541747 final-step override from the previous iteration (no further changes).
3. Direct call to `agent_vitals.backtest._replay_trace` from a script: TP=282, FP=4, F1=0.9495 — bit-identical to bench prototype reference.
4. Full evaluator run (`python -m evaluator.runner --corpus v1 --detectors all --min-confidence 0.8`): identical confab numbers, composite gate PASS.
5. Per-trace diff against prototype reference (from prior iteration's diagnostic): now zero divergent traces in either direction. Strict equivalence.

## Recommendation

**Ship c39ae64 as v1.13.0 M01 release candidate.** No further iterations needed on the confab detector. Bench is at exact parity with the prototype reference and ready to validate M02 (hybrid runaway cost TDA) when that's ready.

## Reference Files

- Bench prototype reference: `prototypes/causal_confab.py:detect_causal_confabulation`
- Agent-vitals integrated detector: `agent_vitals/detection/loop.py:_detect_causal_confabulation` (now matches prototype semantics on every call)
- Agent-vitals replay loop: `agent_vitals/backtest.py:_replay_trace` (final-step override from 8541747)
- Bench mirror with override: `evaluator/runner.py:139` (uncommitted, retained from 8541747 iteration)
- Iteration chain: `reports/eval-agent-vitals-{b3436ef,f6c73f8,78d2896,55dbb79,8541747,c39ae64}-validation.md`
