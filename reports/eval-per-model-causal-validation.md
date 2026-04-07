# Per-Model Causal Confabulation Validation Report

**Date**: 2026-04-06
**Sprint**: 15, Mission s15-m01 (M06 Close-Out)
**Detector**: causal_confab v2 (with verified_source_decoupling path)
**Corpus**: v1/elicited_cross_model, min_confidence >= 0.8

## Summary

The causal confabulation detector was extended with **Path 3: Verified Source
Decoupling** to handle real LLM confabulation traces where `findings_count` and
`sources_count` grow in lockstep but `verified_sources_count` stagnates.

Validated across **11 model families** from 6 providers. The detector achieves
**100% recall on all 3 target models** (gpt-4.1-mini, gemini-2.5-flash,
claude-4-sonnet) with **zero false positives**.

## Target Model Results

| Model | Provider | TP | FP | FN | TN | F1 | P_lb | R_lb |
|-------|----------|----|----|----|----|-----|------|------|
| gpt-4.1-mini | OpenRouter | 12 | 0 | 0 | 0 | 1.000 | 0.757 | 0.757 |
| gemini-2.5-flash | OpenRouter | 9 | 0 | 0 | 3 | 1.000 | 0.701 | 0.701 |
| claude-4-sonnet | OpenRouter | 2 | 0 | 0 | 6 | 1.000 | 0.342 | 0.342 |

**Combined target models**: 23/23 (100%) detection rate, 0 false positives.

## All Model Results

| Model | Provider | TP | FP | FN | TN | F1 | Detection Rate |
|-------|----------|----|----|----|----|-----|----------------|
| gpt-4.1-mini | OpenRouter | 12 | 0 | 0 | 0 | 1.000 | 100% |
| gemini-2.5-flash | OpenRouter | 9 | 0 | 0 | 3 | 1.000 | 100% |
| gpt-4o | OpenAI | 15 | 0 | 0 | 9 | 1.000 | 100% |
| gpt-4o-mini | OpenAI | 8 | 0 | 0 | 0 | 1.000 | 100% |
| deepseek-chat | DeepSeek | 6 | 0 | 0 | 13 | 1.000 | 100% |
| claude-sonnet | Anthropic | 5 | 0 | 0 | 17 | 1.000 | 100% |
| gemini-2.0-flash | OpenRouter | 3 | 0 | 0 | 9 | 1.000 | 100% |
| claude-4-sonnet | OpenRouter | 2 | 0 | 0 | 6 | 1.000 | 100% |
| minimax-m2.7 | MiniMax | 4 | 0 | 1 | 0 | 0.889 | 80% |
| qwen3-coder-480b | OpenRouter | 0 | 0 | 0 | 7 | N/A | N/A (no positives) |

**Total**: 64 TP, 0 FP, 1 FN across 11 models. Aggregate recall: 98.5%.

## Detector Extension: Path 3 (Verified Source Decoupling)

### Problem

The original causal detector (Paths 1-2) tracked `findings_count` ->
`sources_count` coupling. In real LLM elicitation, each cited paper is both a
finding and a source, so findings and sources always grow 1:1. The original
link never breaks, giving 0% detection on elicited traces.

### Solution

Path 3 tracks `verified_sources_count` -> `sources_count` coupling instead.
When a model confabulates, it generates fabricated DOIs that fail verification.
Verified sources stagnate while total sources grow, breaking the verified link.

**Detection criteria:**
- All verified_link_strength values <= 0.5 (persistently weak)
- Final verified/total ratio <= 0.3
- Total sources >= 10

### Impact on Full Corpus

| Metric | Without Path 3 | With Path 3 | Delta |
|--------|---------------|-------------|-------|
| TP | 220 | 282 | +62 |
| FP | 4 | 4 | +0 |
| FN | 88 | 26 | -62 |
| TN | 118 | 118 | 0 |
| F1 | 0.827 | 0.950 | +0.123 |
| P_lb | 0.955 | 0.965 | +0.010 |
| R_lb | 0.661 | 0.879 | +0.218 |

Path 3 recovered 62 true positives with **zero new false positives**.

## Key Finding: Signal Mismatch Between Synthetic and Elicited Traces

Synthetic confabulation traces model confabulation as findings/sources
decoupling (findings grow, sources lag). Real LLM confabulation manifests as
verified/unverified source ratio degradation (both grow, but verified stagnates).
The causal insight (structural decoupling signals confabulation) is the same;
the decoupled signals differ.

## Configuration

```
verified_link_floor: 0.5    # max verified_link_strength must be below this
verified_ratio_gate: 0.3    # final verified/total ratio threshold
verified_min_sources: 10    # minimum total sources for detection
```

## Upstream Implications

The upstream integration spec (`specs/causal-confab-detector.md`) should be
updated to include Path 3. The agent-vitals implementation needs access to
`verified_sources_count` in VitalsSnapshot signals, which is already available
in the schema.
