# Academic Corpus Label Audit — Sprint 22 M01

**Mission**: sprint-22-m01 — Academic Corpus Acquisition + Label Audit
**Date**: 2026-04-21
**Purpose**: Decide per-family evaluation shape for sprint-22-m03 by auditing what ground-truth labels each academic benchmark corpus ships with and whether those labels map onto the 5 agent-vitals detectors (loop / stuck / confabulation / thrash / runaway_cost).

## TL;DR

**All five families land in the same bucket: run-level-outcome-only → flag-frequency analysis.** No family ships per-detector ground truth (no corpus distinguishes loop vs stuck vs confabulation vs thrash vs runaway_cost as the cause of an incident). M03 must therefore produce **composite-classifier** evaluations (does any detector flag predict the run-level incident?), not per-detector confusion matrices.

This changes m03's deliverable shape from what sprint 22 was originally scoped against. See "Decision" section below.

## Scope clarification: 5-family total is 1,500 traces, not 1,917

The metrics team's cross-corpus matrix references "1,917 traces" for the TR thread. That number is the **full** `features_v3_trace_level_enriched.jsonl` row count, which includes:

| Source | Rows |
|---|---|
| AgentProcessBench (4 families: bfcl, gaia_dev, hotpotqa, tau2) | 1,000 |
| CoderForge (SWE-bench Verified trajectories, 2026-04-18 pull) | 500 |
| Pre-pivot SWE rows from `SWE-bench/SWE-smith-trajectories` (via TR-20260417-001 pivot-probe) | 50 |
| InSTA (`data-for-agents/insta-150k-v3`, unlabeled slice) | 250 |
| TRAIL (`PatronusAI/TRAIL`, gaia split) | 117 |
| **Total** | **1,917** |

Per the user's 2026-04-21 scope correction, Sprint 22 is explicitly the 5 families `bfcl + gaia_dev + hotpotqa + tau2 + swe_verified_coderforge`. TRAIL and InSTA are out of scope (they are excluded here for the same reason metrics corrected their draft message: TRAIL is gated, InSTA is unlabeled). The 50 pre-pivot SWE rows are also out of scope because they come from a different trajectory source (`SWE-smith-trajectories`, not `CoderForge`) and were carried forward from an earlier mission for continuity rather than added as new CoderForge data.

**Sprint 22 authoritative count: 1,500 traces** (1,000 APB + 500 CoderForge).

## Per-family label contract

### APB: `bfcl`, `gaia_dev`, `hotpotqa`, `tau2`

**Source**: `https://github.com/RUCBM/AgentProcessBench.git` (staged locally at [corpus/academic/apb/](../corpus/academic/apb/))

**Raw record shape (each line of `<family>.jsonl`)**:
- `question` — user task prompt
- `data_source` — e.g. `bfcl_multi_turn_base`
- `messages` — OpenAI-style transcript: `[{role: system|user|assistant|tool, content, tool_calls?}, ...]`
- `step_labels` — dict mapping assistant-message index → ternary label `{-1, 0, 1}`
- `final_label` — trace-level ternary `{-1, 0, 1}`
- `ground_truth` — task-specific reference answer

**Label semantics** (per APB paper + metrics' TR-20260418-001 binarization convention):
- Step-level ternary: `-1` = process-quality incident (something went wrong at this step), `0` = neutral, `1` = positive step.
- Trace-level binary: `incident = 1` iff **any** step has `step_label < 0` (any-negative-step rule).

**Class balance (raw counts as staged):**

| Family | rows | incident (`any step < 0`) | calm | Incident rate |
|---|---|---|---|---|
| bfcl | 250 | 184 | 66 | 0.736 |
| gaia_dev | 250 | 183 | 67 | 0.732 |
| hotpotqa | 250 | 104 | 146 | 0.416 |
| tau2 | 250 | 143 | 107 | 0.572 |
| **APB total** | **1,000** | **614** | **386** | **0.614** |

**Mapping to agent-vitals detector labels**: **none.** The APB incident label is a process-quality generic flag (the human annotator marked "something went wrong" at a step); it does not distinguish whether the failure was a loop, a stuck pattern, a confabulation, thrash, or runaway cost. The `data_source` field further narrows the task domain (function-calling for bfcl, research-QA for gaia_dev/hotpotqa, multi-turn for tau2) but not the detector category.

**Per-detector confusion matrix: not viable.** The correct evaluation shape is:
- For each detector, compute the fraction of `incident=1` traces where the detector flagged vs. the fraction of `incident=0` traces where it flagged (recall-like vs. false-flag-rate).
- For a composite classifier, compute the fraction of `incident=1` traces where **any** detector flagged vs. same for `incident=0`.

APB does give us a step-level signal (step_labels) that could support a step-level flag-frequency analysis if the detectors are extended to emit per-step flags. That's a bonus path, not a sprint-22 requirement.

### `swe_verified_coderforge`

**Source**: `togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories` (messages + binary reward) joined with `princeton-nlp/SWE-bench_Verified` (issue metadata). Staged locally at [corpus/academic/swe_verified_coderforge/trajectories.jsonl](../corpus/academic/swe_verified_coderforge/trajectories.jsonl) via [scripts/dump_swe_verified_coderforge.py](../scripts/dump_swe_verified_coderforge.py).

**Raw record shape (each line of `trajectories.jsonl`)**:
- `trajectory_id`, `instance_id`, `run_id`
- `reward` — float, binary `{0.0, 1.0}` in this slice (zero non-binary rewards observed)
- `messages` — OpenAI-style transcript from CoderForge-Preview-32B agent run
- `output_patch` — unified-diff patch the agent submitted at end of trajectory
- `verified_issue` — joined metadata: `problem_statement`, `hints_text`, `repo`, `base_commit`, `test_patch`, `patch`

**Label semantics**:
- `reward == 1.0` → all harness tests passed → `incident = 0` (calm).
- `reward == 0.0` → harness failed → `incident = 1` (per metrics convention: failure is the positive class).

**Class balance (raw counts as staged):**

| Family | rows | incident (`reward < 1.0`) | calm | Incident rate |
|---|---|---|---|---|
| swe_verified_coderforge | 500 | 203 | 297 | 0.406 |

**Mapping to agent-vitals detector labels**: **none.** CoderForge reward is a single binary pass/fail signal at the end of the trajectory. No per-step annotation. No per-detector-category annotation.

**Per-detector confusion matrix: not viable.** Same run-level flag-frequency analysis shape as APB families.

## Data-integrity notes

1. **Determinism**: all 5 family artifacts have committed SHA-256 hashes in the per-family `manifest.json`. Re-running `scripts/dump_swe_verified_coderforge.py` against the same HF cache should reproduce byte-identical output.
2. **APB license ambiguity**: GitHub clone has no `LICENSE` file; HF dataset card advertises MIT. metrics proceeded under user override on 2026-04-18 and recorded the ambiguity; we inherit that posture here.
3. **CoderForge license**: no explicit license field exposed via the dataset API at pull time. Unresolved.
4. **APB step-final disagreement**: metrics reported 114 traces (out of 1,000) where the any-negative-step trace label disagrees with a naive `final_label < 0` binarization. Sprint 22 uses the any-negative-step rule (matches metrics convention) and treats the disagreement as a labeling-taxonomy question, not a data-quality issue.

## Decision

**All 5 families → run-level flag-frequency-by-outcome analysis.** Per-family eval reports in M03 will contain:

- For each of the 5 agent-vitals detectors: flag-rate on `incident=1` traces vs. flag-rate on `incident=0` traces, with Wilson 95% CIs on each rate and on their difference.
- A composite `any-detector-fires` classifier: precision / recall / F1 on the run-level `incident` label, with Wilson CIs.
- A confusion-matrix table treating the composite classifier as the predictor and the run-level `incident` label as ground truth. (This is a composite-classifier confusion matrix, not a per-detector one.)

**Not produced in M03**: per-detector confusion matrices on the agent-vitals label set (`loop`, `stuck`, etc.), because no academic family provides that ground truth.

**USS head-to-head** (M03 appendix) follows the same shape: handcrafted composite precision/recall/F1 on `incident` alongside the USS logistic-CV AUC numbers from `features_v3_trace_level_enriched.jsonl`, on the overlapping trace set.

## Open questions for M02

These don't block M01 close-out but shape the adapter work:

1. **Step-ordering / message-indexing convention**: APB `step_labels` keys reference `assistant_message_index` (zero-indexed positions within the `messages` array filtered to `role=="assistant"`). M02 adapter must emit one VitalsSnapshot per assistant step to preserve per-step flag-frequency analysis as a bonus artifact.
2. **Cost / token fields**: neither corpus provides token counts or cost. `runaway_cost` detector will run in degraded mode on all 5 families — log the gap in M02's per-family gap table rather than skipping the detector.
3. **Tool-call richness**: APB has function-call traces (bfcl) and research-QA traces (hotpotqa) with very different tool-call density. M02 must derive `unique_tool_count` / call counts uniformly from `messages[].tool_calls` fields.
