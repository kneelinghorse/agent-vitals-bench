# Integration Spec: Causal Confabulation Detector

**Status:** Ready for implementation
**Source:** agent-vitals-bench Sprint 14 + Sprint 15 (prototypes/causal_confab.py)
**Bench results:** F1=0.995 synthetic, F1=0.950 full corpus with Path 3 (282 TP, 4 FP, 26 FN)
**Replaces:** Current handcrafted SFR-threshold confabulation detector

**Spec version:** 2 (adds Path 3 for elicited LLM traces)
**Last updated:** 2026-04-06

## Summary

Replace the threshold-based source-finding-ratio (SFR) confabulation detector with a causal detector that monitors the structural link between `findings_count` and `sources_count` over rolling windows. The detector fires when the causal link breaks (findings grow without corresponding source growth), which is the invariant mechanism behind confabulation regardless of model or prompt.

## Why This Is Better

The handcrafted SFR detector has two weaknesses the causal approach eliminates:

1. **False positives on delayed-onset traces** (9 FP in bench): SFR drops below threshold temporarily during normal delayed-recovery patterns. The causal detector avoids this because it requires a *structural break* from an established baseline, not just a low ratio.
2. **Threshold brittleness**: SFR thresholds require per-corpus calibration. The causal detector uses relative structural change (baseline vs. current link strength), making it inherently more robust to corpus composition shifts.

## Detection Logic

The detector uses a **three-path rule**. Paths 1-2 catch synthetic confabulation (findings/sources decoupling). Path 3 catches real LLM confabulation (verified/total source decoupling).

### Why three paths?

Synthetic and elicited confabulation produce structurally different signals:
- **Synthetic:** `findings_count` grows while `sources_count` lags -- the primary findings→sources link breaks (Paths 1-2)
- **Elicited (real LLM):** Each fabricated paper is both a finding and a source, so `findings_count == sources_count` always. The decoupling appears between `verified_sources_count` and `sources_count` -- verified stagnates while total grows (Path 3)

The causal insight is the same in both cases: structural decoupling between coupled signals. Only the decoupled signal pair differs.

### Path 1: Causal Link Break (structural break from healthy baseline)

Trigger when ALL of:
- Early baseline link strength >= `baseline_floor` (0.4) -- the trace started with healthy findings-sources coupling
- Weakest window link strength <= `weak_link_threshold` (0.35) -- coupling collapsed later
- Structural drop (baseline - weakest) >= `structural_drop_threshold` (0.2) -- the collapse is significant
- Final source-finding ratio <= `ratio_gate` (0.4) -- the trace ends in a confabulated state

### Path 2: Persistent Low Causal Link (never-established coupling)

Trigger when ALL of:
- Weakest window link strength <= `low_link_threshold` (0.2) -- coupling was never strong
- Initial sources count <= `source_bootstrap_cap` (10) -- started from a small bootstrap
- Final source-finding ratio <= `low_link_ratio_gate` (0.3) -- the trace ends poorly

### Path 3: Verified Source Decoupling (real LLM confabulation)

This path catches confabulation in elicited traces where the model generates fabricated DOIs that fail external verification. Required only if your VitalsSnapshot includes `verified_sources_count`/`unverified_sources_count` fields.

Trigger when ALL of:
- Maximum verified link strength across all windows <= `verified_link_floor` (0.5) -- verified link is *persistently* weak across the entire trace, not just at the end
- Final `verified_sources_count / sources_count` ratio <= `verified_ratio_gate` (0.3) -- most cited sources failed verification
- Final `sources_count` >= `verified_min_sources` (10) -- enough total sources to make the ratio meaningful

**Priority:** Paths 1, 2, 3 are evaluated in order. The first matching path wins. Path 3 only fires when verified data is present and Paths 1-2 do not match.

## Rolling Window Scoring

For each sliding window of `window_size` (4) consecutive snapshots:

### Step 1: Compute deltas
```
findings_deltas = [max(0, findings[i] - findings[i-1]) for each consecutive pair]
sources_deltas  = [max(0, sources[i] - sources[i-1]) for each consecutive pair]
token_deltas    = [max(0, tokens[i] - tokens[i-1]) for each consecutive pair]
```

### Step 2: Compute response ratio
```
response_ratio = sum(sources_deltas) / max(1, sum(findings_deltas))
```

### Step 3: Residualize against token spend (partial correlation)

Remove the confounding effect of token volume on both findings and sources growth:
```
findings_residuals = residualize(findings_deltas, token_deltas)
sources_residuals  = residualize(sources_deltas, token_deltas)
```

Residualization is simple OLS: fit `target = slope * control + intercept`, return residuals. If control variance is zero, return mean-centered target.

### Step 4: Compute partial correlation

Pearson correlation between `findings_residuals` and `sources_residuals`. If either has zero variance, partial correlation is `None`.

### Step 5: Compute link strength
```
if partial_correlation is None:
    link_strength = min(1.0, response_ratio)
else:
    normalized_corr = (partial_correlation + 1.0) / 2.0  # map [-1,1] to [0,1]
    link_strength = max(0.0, min(1.0, normalized_corr * min(1.0, response_ratio)))
```

### Step 6: Select baseline and weakest windows
- **Baseline**: max link_strength of first 2 windows (early trace)
- **Weakest**: min link_strength of windows[1:] (post-baseline)
- **Structural drop**: baseline - weakest (clamped to >= 0)

### Step 7: Verified source link (Path 3 only)

For each window, additionally compute a `verified_link_strength` using the same residualized partial correlation pattern, but with different signals:

```
verified_deltas = [max(0, c - p) for p, c in zip(verified[:-1], verified[1:])]
sources_deltas  = [max(0, c - p) for p, c in zip(sources[:-1], sources[1:])]
token_deltas    = [max(0, c - p) for p, c in zip(tokens[:-1], tokens[1:])]

# Skip windows with no source growth
if sum(sources_deltas) == 0: return None

verified_response  = sum(verified_deltas) / max(1, sum(sources_deltas))
verified_residuals = residualize(verified_deltas, token_deltas)
sources_residuals  = residualize(sources_deltas, token_deltas)
corr               = pearson(verified_residuals, sources_residuals)

if corr is None:
    verified_link_strength = min(1.0, verified_response)
else:
    normalized = (corr + 1.0) / 2.0
    verified_link_strength = max(0.0, min(1.0, normalized * min(1.0, verified_response)))
```

The verified link is `None` when:
- `verified_sources_count` is missing/None on all snapshots in the window
- Total source delta across the window is zero (cannot compute response)

Path 3's "max verified_link_strength <= verified_link_floor" check considers only windows where verified data is present.

## Confidence Scoring

### For `causal_link_break` trigger:
```
baseline_term = min(1.0, baseline_strength / baseline_floor)
drop_term     = min(1.0, structural_drop / structural_drop_threshold)
ratio_term    = 1.0 - min(1.0, final_ratio / ratio_gate)
confidence    = min(1.0, 0.35 + 0.35*baseline_term + 0.2*drop_term + 0.1*ratio_term)
```

### For `persistent_low_causal_link` trigger:
```
low_link_term = 1.0 - min(1.0, weakest_strength / low_link_threshold)
ratio_term    = 1.0 - min(1.0, final_ratio / low_link_ratio_gate)
confidence    = min(1.0, 0.4 + 0.4*low_link_term + 0.2*ratio_term)
```

### For `verified_source_decoupling` trigger (Path 3):
```
weak_term  = 1.0 - min(1.0, verified_weakest / verified_weak_threshold)
ratio_term = 1.0 - min(1.0, final_verified_ratio / verified_ratio_gate)
drop_term  = min(1.0, verified_drop / verified_drop_threshold) if verified_drop > 0 else 0.0
confidence = min(1.0, 0.4 + 0.3*weak_term + 0.15*drop_term + 0.15*ratio_term)
```

Where:
- `verified_baseline` = max verified_link_strength across windows
- `verified_weakest`  = min verified_link_strength across windows
- `verified_drop`     = max(0, verified_baseline - verified_weakest)
- `final_verified_ratio` = `verified_sources_count / sources_count` on the last snapshot

### When not triggered:
```
confidence = min(0.49, 0.2 + 0.2*drift + 0.09*ratio_term)
```

This ensures non-detections never reach the 0.5 confidence threshold.

## Configuration Defaults

| Parameter | Default | Path | Description |
|-----------|---------|------|-------------|
| `window_size` | 4 | All | Sliding window width in snapshots |
| `baseline_floor` | 0.4 | 1 | Minimum link strength to count as "healthy baseline" |
| `weak_link_threshold` | 0.35 | 1 | Below this, link is considered weak |
| `structural_drop_threshold` | 0.2 | 1 | Minimum baseline-to-weakest drop to trigger |
| `ratio_gate` | 0.4 | 1 | Final SFR must be at or below this for Path 1 |
| `low_link_threshold` | 0.2 | 2 | Maximum link strength for Path 2 |
| `source_bootstrap_cap` | 10 | 2 | Maximum initial sources for Path 2 |
| `low_link_ratio_gate` | 0.3 | 2 | Final SFR must be at or below this for Path 2 |
| `verified_link_floor` | 0.5 | 3 | Max verified_link_strength must be below this |
| `verified_weak_threshold` | 0.25 | 3 | Confidence scoring scale for verified weakness |
| `verified_drop_threshold` | 0.15 | 3 | Confidence scoring scale for verified drop |
| `verified_ratio_gate` | 0.3 | 3 | Final verified/total ratio threshold |
| `verified_min_sources` | 10 | 3 | Minimum total sources for Path 3 to fire |

These defaults produce:
- **Synthetic v1 (confab subset):** F1=0.995, P=1.0, R=0.989
- **Full v1 corpus (with Path 3):** F1=0.950, P_lb=0.965, R_lb=0.879 -- 282 TP, 4 FP, 26 FN
- **Elicited cross-model (3 target families):** 23/23 detection, 0 FP via Path 3

Known failure modes:
- 2 FN on "baseline never established" traces with zero early source growth (Paths 1-2)
- 26 FN on full corpus -- mostly very short elicited traces below Path 3's `verified_min_sources` floor

## Required VitalsSnapshot Fields

### Always required (Paths 1-2)
- `signals.findings_count` (int, >= 0)
- `signals.sources_count` (int, >= 0)
- `signals.total_tokens` (int, >= 0)
- `loop_index` (int, >= 0)
- `source_finding_ratio` (Optional[float]) -- used for final ratio if available, else computed from sources/findings

### Required for Path 3 (real LLM confabulation)
- `signals.verified_sources_count` (Optional[int], >= 0)
- `signals.unverified_sources_count` (Optional[int], >= 0)

When `verified_sources_count` is `None` on all snapshots, Path 3 simply does not fire — the detector falls back to Paths 1-2 only. This means systems without DOI verification still get Paths 1-2 detection at full strength, with Path 3 as a transparent upgrade when verification data becomes available.

## Output Schema

```python
@dataclass
class CausalConfabResult:
    detected: bool              # True if confabulation detected
    confidence: float           # 0.0-1.0
    trigger: str | None         # "causal_link_break", "persistent_low_causal_link",
                                # "verified_source_decoupling", or None
    baseline_strength: float    # Link strength of early baseline
    weakest_strength: float     # Lowest link strength observed
    structural_drop: float      # baseline - weakest
    final_ratio: float          # Source-finding ratio at trace end
    initial_sources: int        # sources_count at step 0
    onset_step: int | None      # Loop index where weakness begins
    window_scores: list[...]    # Per-window scoring details (includes verified_link_strength)
```

The `trigger` value tells you which path fired. This is useful for observability and debugging — Path 3 firings indicate real LLM confabulation, Paths 1-2 indicate synthetic-style decoupling.

## Integration Points

### Where it fits in the snapshot pipeline

The causal confab detector should replace the current confabulation detection in the snapshot emission pipeline. It needs the same inputs (the rolling window of recent VitalsSnapshots) and produces the same output interface (`confabulation_detected`, `confabulation_confidence`, `confabulation_trigger`).

### Mapping to existing VitalsSnapshot fields

```
snapshot.confabulation_detected  = result.detected
snapshot.confabulation_confidence = result.confidence
snapshot.confabulation_trigger   = result.trigger
snapshot.confabulation_signals   = [
    f"link_strength={result.weakest_strength:.3f}",
    f"structural_drop={result.structural_drop:.3f}",
    f"baseline={result.baseline_strength:.3f}",
]
```

### Minimum trace length

The detector requires at least `window_size` (4) snapshots. For shorter traces, return `detected=False, confidence=0.0, trigger="insufficient_history"`.

## Reference Implementation

The complete, tested prototype is at `prototypes/causal_confab.py` in the agent-vitals-bench repo. Key functions:
- `detect_causal_confabulation()` -- main entry point
- `score_causal_windows()` -- rolling window scoring
- `_residualize()` / `_pearson()` -- partial correlation helpers

## Bench Validation

After implementing, run the bench evaluator against the v1 corpus to confirm:

### Without Path 3 (Paths 1-2 only)
- Synthetic confab subset: F1 >= 0.990, precision >= 0.98, recall >= 0.96
- Full corpus: confabulation FP count <= 4
- Elicited traces: ~0% recall (expected — Paths 1-2 cannot see verified-source decoupling)

### With Path 3 enabled
- Synthetic confab subset: F1 >= 0.990 (Path 3 should not interfere)
- Full corpus: F1 >= 0.945, P_lb >= 0.96, R_lb >= 0.87
- Elicited cross-model traces: >= 95% recall on traces with `verified_sources_count` data
- Confabulation FP count <= 4 (Path 3 added zero new FP in bench)
- No regression on other detectors (loop, stuck, thrash, runaway_cost)
