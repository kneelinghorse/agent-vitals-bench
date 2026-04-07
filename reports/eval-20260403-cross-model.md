# Cross-Model Detection Analysis

**Generated:** 2026-04-04
**Models:** claude-sonnet (Anthropic), gpt-4o (OpenAI)
**Tier:** frontier
**Traces:** 36 total (30 positive, 6 negative)
**Detectors tested:** confabulation, thrash, runaway_cost

## Per-Model Detection Rates

| Model | Detector | TP | FP | FN | TN | P | R | P_lb | R_lb |
|-------|----------|----|----|----|----|---|---|------|------|
| claude-sonnet | confabulation | 0 | 0 | 5 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| claude-sonnet | thrash | 5 | 0 | 0 | 1 | 1.000 | 1.000 | 0.566 | 0.566 |
| claude-sonnet | runaway_cost | 5 | 0 | 0 | 1 | 1.000 | 1.000 | 0.566 | 0.566 |
| gpt-4o | confabulation | 2 | 0 | 3 | 1 | 1.000 | 0.400 | 0.342 | 0.118 |
| gpt-4o | thrash | 5 | 0 | 0 | 1 | 1.000 | 1.000 | 0.566 | 0.566 |
| gpt-4o | runaway_cost | 5 | 0 | 0 | 1 | 1.000 | 1.000 | 0.566 | 0.566 |

**Note:** Sample sizes (5 positives per model per detector) are below the 25 minimum for formal Wilson CI gate assessment. All formal gates show FAIL due to insufficient sample size, not detection failure. These results are directional findings.

## Cross-Model Generalization Summary

| Detector | claude-sonnet | gpt-4o | Avg Detection Rate | Generalizes? |
|----------|---------------|--------|-------------------|--------------|
| confabulation | 0/5 (0%) | 2/5 (40%) | 20% | **NO** |
| thrash | 5/5 (100%) | 5/5 (100%) | 100% | Yes |
| runaway_cost | 5/5 (100%) | 5/5 (100%) | 100% | Yes |

## Key Findings

1. **thrash** and **runaway_cost** generalize perfectly across frontier models (100% detection rate on both claude-sonnet and gpt-4o). These detectors work on structural patterns (error oscillation, cost burn rate) that are model-independent.

2. **confabulation** does NOT generalize to frontier API models: 0% detection on claude-sonnet, 40% on gpt-4o. The confabulation detector relies on source/finding ratio patterns that differ significantly between model classes. Claude's confabulation patterns are structurally different from those in the synthetic and locally-elicited training corpus.

3. **Model sensitivity:** Confabulation detection is inherently model-sensitive because different models confabulate differently — some produce plausible-looking citations, others fabricate in patterns that the source_finding_ratio heuristic doesn't catch. This aligns with the existing decision that "confabulation is model-aware."

4. **Sample size limitation:** With only 5 positives per model per detector, these findings are directional. A full campaign (20+ per model per detector across 3 tiers) would be needed for publication-grade cross-model CI bounds. The frontier-only run was chosen to stay within session time constraints.

## Implications for Paper C

- **Claim supported:** thrash and runaway_cost detection generalizes across model providers
- **Claim NOT supported:** confabulation detection generalizes across models — Paper C must note this as a limitation
- **Recommendation:** confabulation detector needs model-specific calibration or model-aware thresholds for cross-model deployment
- **Future work:** Extend to mid-range and volume tiers, increase sample sizes to 25+ per model per detector

## Methodology

Traces were elicited by prompting each model with structured scenarios designed to trigger (positive) or avoid (negative) each detector pattern. Elicitation uses the same protocols as the existing corpus but routes through API providers instead of local inference. Each trace is a sequence of VitalsSnapshot objects. Cross-validation runs the detector against each trace and records whether the target pattern was detected.
