# Academic Adapter Signal Coverage Gaps — Sprint 22 M02

**Mission**: sprint-22-m02 — Academic Trace → VitalsSnapshot Adapter Layer
**Date**: 2026-04-21
**Purpose**: Document which agent-vitals signals can and cannot be populated from chat-transcript-only academic corpora, and what that means for detector behaviour on these traces.

## Signal-derivation contract

The academic corpora (APB, CoderForge) ship chat transcripts + run-level labels but **no telemetry**: no findings/sources counts, no coverage scores, no token counts, no error counts. The adapter derives everything from message structure. [adapters/academic/_base.py](../adapters/academic/_base.py) is the single source of truth; family-specific adapters ([apb.py](../adapters/academic/apb.py), [swe_verified_coderforge.py](../adapters/academic/swe_verified_coderforge.py)) are thin format-wrappers that defer to it.

One `VitalsSnapshot` is emitted per **assistant step** (assistant turn + any tool responses that follow before the next assistant turn).

| RawSignals field | Derivation | Confidence |
|---|---|---|
| `findings_count` | cumulative count of assistant turns that produced output (text or tool_calls) | structural-proxy |
| `sources_count` | cumulative count of distinct tool names invoked | structural-proxy |
| `objectives_covered` | same as `findings_count` | structural-proxy |
| `coverage_score` | `min(1.0, findings_count / max(trace_length, 5))` | structural-proxy |
| `confidence_score` | `0.5` (constant — no signal available) | **none** |
| `total_tokens` | cumulative `len(message_text) // 4` across all step text | rough proxy |
| `prompt_tokens` | `0.6 * total_tokens` | rough proxy |
| `completion_tokens` | `total_tokens - prompt_tokens` | rough proxy |
| `api_calls` | same as `findings_count` | structural-proxy |
| `query_count` | same as `findings_count` | structural-proxy |
| `unique_domains` | same as `sources_count` | structural-proxy |
| `refinement_count` | `0` (not inferable) | **none** |
| `convergence_delta` | `0.0` (not inferable) | **none** |
| `error_count` | cumulative matches of `/\b(error|exception|traceback|failed|failure|invalid|cannot|could not)\b/i` in tool-response text | heuristic |
| `verified_sources_count` / `unverified_sources_count` | `None` (not inferable) | **none** |

Temporal metrics (`cv_coverage`, `cv_findings_rate`, `dm_coverage`, `dm_findings`, `qpf_tokens`, `cs_effort`) are computed from the running series via `agent_vitals.detection.metrics.TemporalMetrics` using the same formulas `agent_vitals.monitor._process_signals` uses in production, so the temporal-metric path is faithful even if the underlying signals are derived.

`output_fingerprint` is computed per step from the assistant text; `output_similarity` is left as `None` (we don't pre-compute inter-step similarity — the detectors fall through to other signals).

## Full-corpus smoke results (2026-04-21, 1,500 traces, 81 s)

Zero adapter-side exceptions. Detector flag rates by family (calm incident-rate / incident flag-rate):

| Family | traces | loop | stuck | confabulation | thrash | runaway_cost |
|---|---|---|---|---|---|---|
| bfcl | 250 (66 calm / 184 incident) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.08 | 0.38 / 0.56 | 0.77 / 0.77 |
| gaia_dev | 250 (67 / 183) | 0.00 / 0.00 | 0.00 / 0.00 | 0.04 / 0.38 | 0.27 / 0.50 | 0.10 / 0.34 |
| hotpotqa | 250 (146 / 104) | 0.00 / 0.00 | 0.00 / 0.00 | 0.17 / 0.39 | 0.04 / 0.12 | 0.00 / 0.04 |
| tau2 | 250 (107 / 143) | 0.00 / 0.00 | 0.00 / 0.00 | 0.64 / 0.60 | 0.07 / 0.46 | 0.79 / 0.69 |
| swe_verified_coderforge | 500 (297 / 203) | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.89 / 0.90 |

Discrimination (incident-rate minus calm-rate, positive = signal):

| Family | loop | stuck | confab | thrash | runaway_cost |
|---|---|---|---|---|---|
| bfcl | 0 | 0 | +0.08 | +0.18 | 0 |
| gaia_dev | 0 | 0 | +0.34 | +0.23 | +0.24 |
| hotpotqa | 0 | 0 | +0.22 | +0.08 | +0.04 |
| tau2 | 0 | 0 | -0.04 | +0.39 | -0.10 |
| swe_verified_coderforge | 0 | 0 | 0 (saturated) | 0 (saturated) | +0.01 |

## Per-detector gap analysis

### `loop` — **silent on all 1,500 traces**
**Why**: loop detection leans on `output_similarity` between consecutive snapshots (CUSUM on output fingerprint diversity, similarity-threshold trips). The adapter leaves `output_similarity=None`; computing pairwise similarity for every assistant message in a 78-message CoderForge trajectory is expensive and wasn't in scope for M02. Loop detection on these corpora is effectively disabled.

**Implication for M03**: report as "not evaluable on academic corpora under the current adapter"; do not report a precision/recall number that looks like zero performance when it's really zero signal.

### `stuck` — **silent on all 1,500 traces**
**Why**: the derived `coverage_score` rises monotonically when the assistant keeps producing output, so `dm_coverage` stays positive and the stuck thresholds never fire. Real stuck traces have actual coverage plateaus; our derivation cannot see a plateau because we increment findings on every assistant turn that emits anything.

**Implication for M03**: same as loop — not evaluable under the current adapter, not a detector failure.

### `confabulation` — **mixed signal, heavily family-dependent**
**Why**: `confabulation` relies primarily on `verified_source_ratio` (Path B) and `source_finding_ratio` (Path A). The adapter sets `verified_sources_count=None` (no academic corpus has citation verification), so Path B is disabled. Path A uses our `sources_count` proxy (distinct tool names) divided by `findings_count`; early in a trace when tools haven't been invoked yet, the ratio looks low and the floor threshold (0.3) fires.

- **gaia_dev** and **hotpotqa** show the best signal (+0.34 / +0.22), because incident traces really do tend to use fewer distinct tools before failing.
- **tau2** shows negative discrimination (-0.04), suggesting the structural proxy is anti-correlated with actual process quality in that workflow.
- **SWE** saturates at 1.00 / 1.00 — CoderForge trajectories use many repeated tool calls, so the ratio is perpetually below floor; the detector fires on every trace and provides no signal.

**Implication for M03**: report per-family but annotate that confabulation flag-rate on academic corpora is a proxy-artefact as much as a detection signal.

### `thrash` — **discriminates on APB, saturates on SWE**
**Why**: `thrash` triggers on CV spikes in findings/coverage/token series. Our derived series have enough variation from real message-length variation to generate plausible CV values. **thrash is the single most useful detector on academic corpora** — discrimination ranges from +0.08 (hotpotqa) to +0.39 (tau2), with no anti-signals. On SWE it saturates (1.00 / 1.00) because trajectories are long enough that CV crosses threshold universally.

**Implication for M03**: thrash-based run-level incident prediction is the one meaningful composite-classifier input on APB. SWE is dominated by saturation.

### `runaway_cost` — **dominated by token-proxy artefacts**
**Why**: our token proxy is cumulative `char_count // 4`, which grows monotonically and roughly linearly with trace length. Short traces don't trip the burn-rate threshold; long ones do. This correlates with incident rate mostly because incident traces tend to be longer (the agent kept trying after a failure).

- **gaia_dev** shows modest real discrimination (+0.24).
- **tau2** is anti-signal (-0.10) because calm tau2 traces are also long.
- **SWE** saturates (0.89 / 0.90) because trajectories are uniformly long.

**Implication for M03**: runaway_cost on academic corpora is not evaluating "is the agent burning tokens wastefully?"; it's mostly evaluating trace length. Flag the confound explicitly.

## Structural gaps (what M03 cannot claim)

1. **No per-detector ground truth** — established in M01, unchanged by M02. All academic-corpus results are run-level flag-frequency only.
2. **`loop` and `stuck` are unevaluable** on academic corpora under the current adapter. Not a detector failure; a signal-availability failure. M03 should either (a) report them as "N/A (no signal)" or (b) skip them from the composite-classifier entirely and document the exclusion.
3. **`runaway_cost` confounded with trace length** on all 5 families. Report the raw number but annotate.
4. **SWE/CoderForge detectors near-saturated** for confab and thrash. The composite any-detector-fires on SWE will be near-1.00 / near-1.00 regardless of incident label — meaningful F1 requires dropping saturated detectors or using a per-detector thresholded score instead.
5. **No cost/token data** — CoderForge ships message text only; APB ships no token counts. `runaway_cost` always runs in proxy-only mode. If metrics can stage the raw CoderForge rollout metadata (prompt/completion token counts per step) we could tighten this; that's an M03 follow-up question, not an M02 blocker.

## Scope note — SWE incident counts

The SWE manifest shows `incident=203, calm=297` against the `reward<1.0 → incident=1` convention, not the `incident=216, calm=334` the metrics TR-20260418-001 data-audit reports. The difference is that the metrics audit's 550-row swe_bench slice concatenates 500 CoderForge trajectories with 50 pre-pivot SWE-smith-trajectory rows (see [academic-corpus-label-audit.md](academic-corpus-label-audit.md)). Sprint 22 evaluates CoderForge-only (500 rows), matching the 2026-04-21 scope correction.

## Open M03 questions (not adapter bugs, not blockers)

1. How does M03 present composite F1 when `loop` and `stuck` are guaranteed zero? Proposed: composite-excluding-silent-detectors, with a footnote listing the exclusions.
2. Does the USS head-to-head appendix need to use the same exclusion policy, or compare against USS's own 8-feature suite as-is? Proposed: USS-suite-as-is (no changes to the TR-thread pipeline) to preserve apples-to-apples with metrics' numbers.
3. Is per-step flag-frequency analysis worth building from APB step_labels? The data is there (`step_labels` in raw records); the adapter preserves snapshot ordering so a future extension could emit per-step predictions. Out of M03 scope unless the run-level analysis is uninformative.
