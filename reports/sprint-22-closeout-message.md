# Sprint 22 Closeout — Message to metrics_and_protocols

**Target**: `cmos://darryl/metrics-and-protocols`
**Type**: `status_update`
**Date drafted**: 2026-04-21
**Status**: BLOCKED ON SEND — dashboard rejected MCP send with "senderAddress required when senderProjectId is provided"; tool interface does not expose senderAddress. Send manually via dashboard UI using the body below until the cmos-mcp tool catches up with the server-side validation change.

**Summary (for inbox list)**: Sprint 22 closeout: cross-corpus evaluation as scoped is not testable on APB/CoderForge; gap analysis + 1 salvageable mini-result attached; recommended path is telemetry-carrying corpus regeneration

**Evidence references**:
- docs/academic-corpus-label-audit.md (sprint-22-m01 output)
- docs/academic-adapter-gaps.md (sprint-22-m02 output)
- sprint-22-m01 mission record
- sprint-22-m02 mission record
- sprint-22-m03 mission record (deferred)

---

## Body

Closing out Sprint 22 on the evidence-gap request you raised on 2026-04-21. Two of three missions landed (M01 corpus acquisition + label audit, M02 adapter layer). M03 (the gate-evaluation run) is being deferred rather than executed — the write-up below explains why, and what we found that's still worth knowing.

### Sprint scope recap

You asked bench to run the handcrafted detectors (via replay_trace() S07/S10) against the 5 academic families used in the USS TR thread (bfcl, gaia_dev, hotpotqa, tau2 from APB + swe_verified_coderforge), to test three interpretive frames: composite F1 >= 0.8 = strong generalization, ~0.70 = corpus ceiling, < USS = handcrafted overfit to bench v1. The authoritative corpus size is 1,500 traces (1,000 APB + 500 CoderForge); the 1,917 number in the cross-corpus matrix includes 250 unlabeled InSTA + 117 out-of-scope TRAIL + 50 pre-pivot SWE-smith rows.

### What M01 + M02 actually found

**Label-mapping finding (M01)**: No academic family ships per-detector ground truth. APB's step/final labels are process-quality ternary flags; CoderForge reward is binary run-pass/fail. Nothing distinguishes loop vs stuck vs confabulation vs thrash vs runaway_cost as the cause of an incident. Per-detector confusion matrices are therefore not viable on any family. M03 was re-scoped to composite-any-detector-fires classifier shape. Full audit at docs/academic-corpus-label-audit.md.

**Signal-availability finding (M02)**: This is the blocker. Academic corpora ship chat messages + labels only, no agent-runtime telemetry. The adapter at adapters/academic/_base.py synthesizes RawSignals from message structure (findings_count = assistant turns that produced output, sources_count = distinct tool names, total_tokens = char_count // 4, error_count = regex heuristic on tool responses, etc.) and computes temporal metrics via agent_vitals.detection.metrics.TemporalMetrics using the production formulas. Full-corpus smoke: 1,500 traces through replay_trace() in 81 seconds, zero adapter-side exceptions. Detector flag rates across families, however:

- `loop`: 0/1500 flags. Requires output_similarity signal we don't pre-compute. Structurally silent.
- `stuck`: 0/1500 flags. Derived coverage monotonically rises when the assistant keeps producing output, so no plateau ever triggers. Structurally silent.
- `confabulation`: mixed. Real signal on gaia_dev (+0.34) and hotpotqa (+0.22); anti-signal on tau2 (-0.04); saturated at 1.00/1.00 on SWE.
- `thrash`: real signal on APB (+0.08 to +0.39); saturated at 1.00/1.00 on SWE.
- `runaway_cost`: confounded with trace length on all 5 families (our token proxy grows monotonically with character count).

Per-family table + discrimination deltas are in docs/academic-adapter-gaps.md.

### Why M03 is deferred, not executed

Any composite F1 M03 would produce is a function of our proxy-derivation choices at least as much as actual detector behavior. If we report F1 = 0.4, someone will quote it as "handcrafted detectors fail out-of-distribution," which is not what we would have measured — we would have measured "detectors fed proxy signals produce proxy-dependent outputs." This doesn't cleanly answer any of your three interpretive frames; worse, it produces a number that reads as an answer while contaminating the actual generalization question.

### The one salvageable mini-result

**Thrash detector discriminates incident from calm on 2 of 4 APB families, with 95% Wilson CIs non-overlapping between incident and calm flag rates**:

| Family | incident flag-rate [CI] | calm flag-rate [CI] | disc | CIs non-overlapping? |
|---|---|---|---|---|
| bfcl | 0.560 [0.488, 0.630] | 0.379 [0.271, 0.499] | +0.18 | no |
| gaia_dev | 0.497 [0.426, 0.569] | 0.269 [0.177, 0.385] | +0.23 | **yes** |
| hotpotqa | 0.125 [0.075, 0.202] | 0.041 [0.019, 0.087] | +0.08 | no |
| tau2 | 0.462 [0.382, 0.543] | 0.075 [0.038, 0.141] | +0.39 | **yes** |

Narrow claim: the thrash detector's CV-based logic, when fed structural proxies from chat transcripts alone (no real telemetry), produces incident/calm discrimination with statistically meaningful separation on two independent task families (gaia_dev research-QA, tau2 multi-turn service). This is a much smaller claim than "handcrafted detectors generalize," but it's actually supportable with what we have. All four APB families show a positive discrimination direction; the two with sufficient effect size pass a Wilson non-overlap check; the two that don't (bfcl, hotpotqa) would need larger n to rule in or out.

SWE contributes nothing to this claim — confab and thrash both saturate at 1.00/1.00 regardless of incident label, runaway_cost is length-confounded, loop/stuck are silent. SWE composite-classifier is not evaluable under the current adapter.

### Recommended path to the original question

To actually answer "do handcrafted detectors generalize to OOD data?", the academic corpora need to carry agent-runtime telemetry. Three options in decreasing order of feasibility:

1. Regenerate equivalent task families locally with `agent_vitals.monitor.VitalsMonitor` wrapping the agent loop, capturing real findings/coverage/token/confidence values alongside the transcript. Bench's elicitation/ modules (elicit_confabulation, elicit_thrash, elicit_runaway_cost) are already this shape — they drive a real LLM through a protocol with the Monitor attached. 250-per-family scale is feasible on local llama.cpp + API providers over roughly a week of wall time. **This is the plan bench is pursuing next.**
2. Ask APB/CoderForge maintainers to re-release with telemetry sidecars. Unrealistic — not their use case, not their labeling budget.
3. Run pairwise output-similarity off the transcripts to unblock the loop detector and invest in smarter tokenization for runaway_cost. This would partially rescue the current adapter but still wouldn't recover coverage/confidence/verified-sources signals — floor is still limited.

### Deliverables landed this sprint

- `corpus/academic/apb/` (4 JSONL families, 1,000 traces, 38 MB, SHA-256 in manifests) — staged from your TR-20260418-001 tmp clone
- `corpus/academic/swe_verified_coderforge/trajectories.jsonl` (500 traces, 130 MB) — dumped offline via `scripts/dump_swe_verified_coderforge.py` against HF cache
- `adapters/academic/` (_base.py + 2 family wrappers) — shared chat-transcript → VitalsSnapshot derivation
- `tests/adapters/academic/` (18 tests, all green)
- `docs/academic-corpus-label-audit.md` (M01 output)
- `docs/academic-adapter-gaps.md` (M02 output, full per-family flag-rate analysis)

The corpus and adapters are permanent infrastructure even though M03 deferred — if a future sprint produces or receives instrumented versions of these task families, they drop in directly.

### Asks

None, unless you want to push back on the defer-M03 decision. If you'd prefer I run M03 with heavy caveats anyway and produce the composite F1 numbers, say so and I'll revise. Otherwise bench is starting on the local-capture path next.
