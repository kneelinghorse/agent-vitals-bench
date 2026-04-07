# Integration Spec: Hybrid Runaway Cost Detector

**Status:** Ready for implementation
**Source:** agent-vitals-bench Sprint 14 (prototypes/causal_runaway.py, prototypes/tda_detector.py)
**Bench results:** TDA F1=1.000 (P_lb=0.981, R_lb=0.981), Handcrafted F1=0.959, Causal F1=0.936
**Approach:** Two-layer hybrid -- handcrafted first-pass screening + TDA adjudication layer

## Summary

The current handcrafted runaway cost detector produces 17 false positives on elicited traces (burn-rate threshold too aggressive on real-world cost patterns). The TDA detector achieves perfect F1=1.000 with zero FP on the aligned subset, but requires a Python 3.12 environment with giotto-tda and drops 20% of short traces.

The recommended integration is a **hybrid architecture**:

1. **First pass (cheap, always available):** Handcrafted burn-rate detector screens all traces. If it says "not runaway," trust it (recall=1.0, zero FN).
2. **Adjudication layer (expensive, optional):** When the handcrafted detector triggers, pass the trace through TDA for confirmation. TDA's zero FP rate eliminates the 17 false alarms.

This gives the system full recall from handcrafted + full precision from TDA.

## Architecture

```
Incoming trace
    |
    v
[Handcrafted Burn-Rate Check] -- not triggered --> healthy (done)
    |
    triggered
    |
    v
[TDA Available?] -- no --> report handcrafted result (accept FP risk)
    |
    yes
    |
    v
[TDA Runaway Classifier] -- confirmed --> runaway_cost detected
    |                     -- not confirmed --> override to healthy
```

### Why this ordering?

- Handcrafted has **perfect recall** (FN=0) but 17 FP -- it never misses a real runaway
- TDA has **perfect precision** (FP=0) but requires optional dependencies and drops short traces
- By using handcrafted as the first filter, we never miss a real runaway
- By using TDA as the adjudicator, we eliminate false alarms when TDA is available
- When TDA is unavailable (missing deps or short trace), the handcrafted result stands

## Layer 1: Handcrafted Burn-Rate Detector (existing)

This is the current production detector -- no changes needed. It serves as the recall-optimized first pass.

Key behavior to preserve:
- `burn_rate_multiplier` threshold-based detection
- Perfect recall on the v1 corpus (FN=0)
- Will produce some FP on elicited/real traces (acceptable, TDA catches these)

## Layer 2: TDA Adjudication

When the handcrafted detector fires, extract TDA features and run the trained GradientBoosting classifier to confirm or override.

### TDA Feature Extraction

The TDA pipeline extracts 663 features per trace using persistent homology:

#### Input
A sequence of VitalsSnapshot objects. Each snapshot contributes a 17-dimensional feature vector:

**Signal features (11):**
- `findings_count`, `sources_count`, `objectives_covered`, `coverage_score`, `total_tokens`
- `error_count`, `confidence_score`, `convergence_delta`
- `prompt_tokens`, `completion_tokens`, `refinement_count`

**Metric features (6):**
- `dm_coverage`, `dm_findings`, `cv_coverage`, `cv_findings_rate`
- `qpf_tokens`, `cs_effort`

#### Step 1: Normalize trace matrix
```python
# trace_matrix shape: (n_steps, 17)
mins = trace_matrix.min(axis=0)
ranges = trace_matrix.max(axis=0) - mins
ranges[ranges == 0] = 1.0
trace_norm = (trace_matrix - mins) / ranges
```

#### Step 2: Build sliding-window point clouds
For each window size in `[3, 4, 5]`:
```python
# Flatten consecutive windows into high-dimensional points
points = [trace_norm[i:i+w].flatten() for i in range(n_steps - w + 1)]
point_cloud = np.array(points)  # shape: (n_steps - w + 1, w * 17)
```

#### Step 3: Compute persistence diagrams
Using Vietoris-Rips persistence (giotto-tda):
```python
vrp = VietorisRipsPersistence(
    homology_dimensions=(0, 1),
    max_edge_length=np.inf,
    n_jobs=1,
)
diagrams = vrp.fit_transform(point_cloud[np.newaxis, :, :])
```

#### Step 4: Extract topological features per window size

For each window size, extract these feature groups from the persistence diagrams:
- **PersistenceEntropy** -- information-theoretic summary
- **Amplitude** (wasserstein, landscape, bottleneck) -- distance-based summaries
- **NumberOfPoints** -- topological feature count
- **PersistenceLandscape** (3 layers, 20 bins) -- functional summaries
- **BettiCurve** (20 bins) -- Betti number evolution
- **Silhouette** (20 bins) -- weighted lifetime profiles
- **Lifetime statistics** per homology dimension (H0, H1): mean, std, max, median, skewness, n_long (count above 75th percentile)

#### Step 5: Concatenate
All features across all 3 window sizes are concatenated into a single 663-dimensional vector, sorted by feature name.

### Minimum trace length

Traces with fewer than `min_steps` (5) snapshots are excluded from TDA processing. When a short trace triggers the handcrafted detector, the handcrafted result stands without TDA adjudication.

The 20% short-trace dropout rate (299/1494 traces in v1) is a validated requirement -- relaxing it degrades confabulation detection quality. Accept this coverage gap.

### Trained Model Artifact

The classifier is a `sklearn.pipeline.Pipeline`:
1. `StandardScaler` -- zero-mean, unit-variance normalization
2. `GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)`

Pre-trained artifacts are saved as joblib files in `prototypes/tda_models/runaway_cost.joblib`. The artifact contains:
```python
{
    "detector": "runaway_cost",
    "pipeline": <fitted Pipeline>,
    "feature_names": ["tda_feature_0", ..., "tda_feature_662"],
    "config": { ... TDAConfig as dict ... },
}
```

### Prediction
```python
X = np.array([feature_vector])  # shape: (1, 663)
pipeline = artifact["pipeline"]
probability = float(pipeline.predict_proba(X)[0][1])
detected = probability >= 0.5
```

## Hybrid Decision Logic

```python
def detect_runaway_cost_hybrid(
    snapshots: list[VitalsSnapshot],
    *,
    tda_available: bool = False,
    tda_model_dir: Path | None = None,
) -> RunawayResult:
    # Layer 1: Handcrafted (always runs)
    handcrafted = detect_runaway_handcrafted(snapshots)

    if not handcrafted.detected:
        # Handcrafted says healthy -- trust it (recall=1.0)
        return handcrafted

    # Layer 2: TDA adjudication (when available)
    if tda_available and len(snapshots) >= 5:
        try:
            tda_features = extract_tda_features(snapshots)
            if tda_features is not None:
                tda_result = predict_runaway(tda_features, model_dir=tda_model_dir)
                if not tda_result.detected:
                    # TDA overrides handcrafted FP
                    return RunawayResult(
                        detected=False,
                        confidence=handcrafted.confidence * 0.3,  # downweight
                        trigger="handcrafted_overridden_by_tda",
                        ...
                    )
                # Both agree -- high confidence
                return RunawayResult(
                    detected=True,
                    confidence=max(handcrafted.confidence, tda_result.probability),
                    trigger="confirmed_by_tda",
                    ...
                )
        except (ImportError, FileNotFoundError):
            pass  # TDA unavailable, fall through

    # TDA unavailable or short trace -- handcrafted result stands
    return handcrafted
```

## Configuration

| Parameter | Default | Layer | Description |
|-----------|---------|-------|-------------|
| `burn_rate_multiplier` | existing | Handcrafted | Current production threshold |
| `min_steps` | 5 | TDA | Minimum trace length for TDA processing |
| `window_sizes` | (3, 4, 5) | TDA | Sliding window sizes for point cloud construction |
| `homology_dimensions` | (0, 1) | TDA | Topological dimensions to compute |
| `tda_available` | false | Hybrid | Feature flag for TDA adjudication |

## Dependencies

### Layer 1 (Handcrafted)
No new dependencies.

### Layer 2 (TDA) -- all optional
- `giotto-tda >= 0.6.0` (requires Python <= 3.12)
- `scikit-learn >= 1.3.0`
- `joblib >= 1.3.0`
- `numpy >= 1.24.0`

TDA dependencies must remain **optional**. The system must function correctly with only Layer 1 when TDA packages are not installed. Use a feature flag or lazy import pattern.

## Output Schema

The hybrid detector should produce the same VitalsSnapshot fields as the current detector:

```
snapshot.runaway_cost_detected    = result.detected
snapshot.runaway_cost_confidence  = result.confidence
snapshot.runaway_cost_trigger     = result.trigger
```

Additional diagnostic fields for observability:
```
snapshot.runaway_cost_signals = [
    f"layer=handcrafted|tda|hybrid",
    f"handcrafted_triggered={handcrafted.detected}",
    f"tda_confirmed={tda_confirmed}",  # None if TDA unavailable
]
```

## Bench Validation

After implementing, run the bench evaluator against the v1 corpus:
- Runaway-only subset: F1 >= 0.995 (hybrid should match or exceed TDA's 1.000 on eligible traces)
- Full corpus: runaway_cost FP count <= 2
- No regression on other detectors
- Verify graceful degradation when TDA dependencies are missing (should fall back to handcrafted-only)

## Model Retraining

The TDA model should be retrained when:
- The corpus grows significantly (> 20% new traces)
- New failure modes are added
- VitalsSnapshot schema changes (signal/metric field additions)

Retraining requires the Python 3.12 TDA environment. Script: `prototypes/tda_detector.py:train_tda_models()`.

## Reference Implementations

- Handcrafted runaway detector: existing production code in agent-vitals
- Causal runaway prototype: `prototypes/causal_runaway.py` (bench reference, not recommended for upstream -- 27 FP from cost-output decoupling on stuck traces)
- TDA detector prototype: `prototypes/tda_detector.py` (bench reference)
- Comparative benchmark report: `reports/eval-causal-tda-comparison.md`
