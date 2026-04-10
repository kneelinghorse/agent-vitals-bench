# Five-Paradigm Comparative Report

**Generated:** 2026-04-07T18:43:46Z
**Corpus:** v1
**Min confidence:** 0.80
**Filtered traces:** 1494
**Detectors:** loop, stuck, confabulation, thrash, runaway_cost (canonical bench five)
**Paradigms:** handcrafted, causal, tda, mamba, hopfield (handcrafted = production rules; causal = rolling causal-window prototype; TDA = persistence-feature gradient boosting; Mamba = state-space sequence model; Hopfield = modern Hopfield network with learnable stored patterns)
**Cutoffs:** 3 steps, 5 steps, 7 steps, Full (early-detection prefixes plus the full trace baseline)

## TL;DR

- **Full-trace per-detector winners:** loop → handcrafted (F1 1.000), stuck → mamba (F1 0.989), confabulation → mamba (F1 1.000), thrash → handcrafted (F1 1.000), runaway_cost → mamba (F1 1.000)
- **Distinct winning paradigms (full trace):** handcrafted, mamba
- **Best macro-F1 across detectors (full trace):** mamba (0.998)

## Methodology

Every paradigm is invoked through the bench M04 partial-trace harness (`evaluator/partial_trace.py:evaluate_at_cutoff`) via a `TracePredictor` adapter (`prototypes/predictor_adapters.py`). This guarantees identical input plumbing — the same manifest filtering, the same trace truncation, the same confusion-matrix accounting — across all 5 paradigms and all 4 cutoffs.

**Standalone semantics, not hybrid.** Each paradigm scores on its own. When a paradigm cannot answer (TDA window-floor, Mamba/Hopfield `min_steps`, causal insufficient history), the affected detectors return `False` rather than falling back to handcrafted. The harness records the full corpus size at every cell, so eligibility gaps register as recall drops rather than being hidden behind a hybrid backstop. This is the only framing that meaningfully distinguishes paradigms at small cutoffs.

**Hopfield uses prefix-trained models.** At cutoff 3 / 5 / 7 the harness loads `{detector}_p3.pt / _p5.pt / _p7.pt`; at full trace it loads `{detector}.pt`. The other paradigms use a single model across cutoffs.

**Corpus.** v1, min_confidence=0.80, 1494 traces. Per-detector positive counts: loop=212, stuck=138, confabulation=308, thrash=261, runaway_cost=229.

## Full-Trace Comparison (5 paradigms × 5 detectors)

### F1

| Paradigm | loop | stuck | confabulation | thrash | runaway_cost |
|---|---|---|---|---|---|
| handcrafted | 1.000 | 0.986 | 0.949 | 1.000 | 0.898 |
| causal | 0.000 | 0.000 | 0.949 | 0.000 | 0.944 |
| tda | 1.000 | 0.871 | 0.818 | 0.939 | 0.930 |
| mamba | 1.000 | 0.989 | 1.000 | 1.000 | 1.000 |
| hopfield | 1.000 | 0.989 | 0.973 | 1.000 | 1.000 |

### Precision lower bound (Wilson 95%)

| Paradigm | loop | stuck | confabulation | thrash | runaway_cost |
|---|---|---|---|---|---|
| handcrafted | 0.982 | 0.930 | 0.965 | 0.985 | 0.765 |
| causal | 0.000 | 0.000 | 0.965 | 0.000 | 0.851 |
| tda | 0.982 | 0.936 | 0.982 | 0.984 | 0.981 |
| mamba | 0.982 | 0.939 | 0.988 | 0.985 | 0.984 |
| hopfield | 0.982 | 0.939 | 0.921 | 0.985 | 0.984 |

### Recall lower bound (Wilson 95%)

| Paradigm | loop | stuck | confabulation | thrash | runaway_cost |
|---|---|---|---|---|---|
| handcrafted | 0.982 | 0.973 | 0.879 | 0.985 | 0.984 |
| causal | 0.000 | 0.000 | 0.879 | 0.000 | 0.984 |
| tda | 0.982 | 0.707 | 0.638 | 0.841 | 0.819 |
| mamba | 0.982 | 0.973 | 0.988 | 0.985 | 0.984 |
| hopfield | 0.982 | 0.973 | 0.982 | 0.985 | 0.984 |

## Per-Detector Winners — Full Trace

| Detector | Winner | F1 | P_lb | R_lb | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| loop | handcrafted | 1.000 | 0.982 | 0.982 | 212 | 0 | 0 |
| stuck | mamba | 0.989 | 0.939 | 0.973 | 138 | 3 | 0 |
| confabulation | mamba | 1.000 | 0.988 | 0.988 | 308 | 0 | 0 |
| thrash | handcrafted | 1.000 | 0.985 | 0.985 | 261 | 0 | 0 |
| runaway_cost | mamba | 1.000 | 0.984 | 0.984 | 229 | 0 | 0 |

## Early-Detection F1 Curves

One table per detector. Rows are paradigms, columns are step-prefix cutoffs plus the full-trace baseline. Cells are F1 on the entire filtered corpus at that cutoff under standalone semantics.

### loop

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.507 | 0.785 | 0.916 | 1.000 |
| causal | 0.000 | 0.000 | 0.000 | 0.000 |
| tda | 0.000 | 0.000 | 0.662 | 1.000 |
| mamba | 0.000 | 0.785 | 0.916 | 1.000 |
| hopfield | 0.936 | 0.986 | 0.960 | 1.000 |

### stuck

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.000 | 0.881 | 0.986 | 0.986 |
| causal | 0.000 | 0.000 | 0.000 | 0.000 |
| tda | 0.000 | 0.000 | 0.566 | 0.871 |
| mamba | 0.000 | 0.525 | 0.989 | 0.989 |
| hopfield | 0.890 | 0.899 | 0.871 | 0.989 |

### confabulation

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.440 | 0.803 | 0.920 | 0.949 |
| causal | 0.000 | 0.803 | 0.920 | 0.949 |
| tda | 0.000 | 0.000 | 0.311 | 0.818 |
| mamba | 0.000 | 0.826 | 0.989 | 1.000 |
| hopfield | 0.901 | 0.920 | 0.818 | 0.973 |

### thrash

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.885 | 1.000 | 1.000 | 1.000 |
| causal | 0.000 | 0.000 | 0.000 | 0.000 |
| tda | 0.000 | 0.000 | 0.583 | 0.939 |
| mamba | 0.000 | 0.695 | 0.966 | 1.000 |
| hopfield | 0.941 | 1.000 | 0.939 | 1.000 |

### runaway_cost

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.497 | 0.956 | 0.909 | 0.898 |
| causal | 0.000 | 0.285 | 0.947 | 0.944 |
| tda | 0.000 | 0.000 | 0.534 | 0.930 |
| mamba | 0.000 | 0.382 | 1.000 | 1.000 |
| hopfield | 0.836 | 1.000 | 0.930 | 1.000 |

## Per-Cutoff Winners

### 3 steps

| Detector | Winner | F1 | P_lb | R_lb | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| loop | hopfield | 0.936 | 0.833 | 0.982 | 212 | 29 | 0 |
| stuck | hopfield | 0.890 | 0.736 | 0.973 | 138 | 34 | 0 |
| confabulation | hopfield | 0.901 | 0.798 | 0.945 | 299 | 57 | 9 |
| thrash | hopfield | 0.941 | 0.847 | 0.985 | 261 | 33 | 0 |
| runaway_cost | hopfield | 0.836 | 0.666 | 0.984 | 229 | 90 | 0 |

### 5 steps

| Detector | Winner | F1 | P_lb | R_lb | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| loop | hopfield | 0.986 | 0.982 | 0.940 | 206 | 0 | 6 |
| stuck | hopfield | 0.899 | 0.802 | 0.881 | 129 | 20 | 9 |
| confabulation | hopfield | 0.920 | 0.850 | 0.925 | 294 | 37 | 14 |
| thrash | handcrafted | 1.000 | 0.985 | 0.985 | 261 | 0 | 0 |
| runaway_cost | hopfield | 1.000 | 0.984 | 0.984 | 229 | 0 | 0 |

### 7 steps

| Detector | Winner | F1 | P_lb | R_lb | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| loop | hopfield | 0.960 | 0.927 | 0.921 | 203 | 8 | 9 |
| stuck | mamba | 0.989 | 0.939 | 0.973 | 138 | 3 | 0 |
| confabulation | mamba | 0.989 | 0.955 | 0.988 | 308 | 7 | 0 |
| thrash | handcrafted | 1.000 | 0.985 | 0.985 | 261 | 0 | 0 |
| runaway_cost | mamba | 1.000 | 0.984 | 0.984 | 229 | 0 | 0 |

### Full

| Detector | Winner | F1 | P_lb | R_lb | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| loop | handcrafted | 1.000 | 0.982 | 0.982 | 212 | 0 | 0 |
| stuck | mamba | 0.989 | 0.939 | 0.973 | 138 | 3 | 0 |
| confabulation | mamba | 1.000 | 0.988 | 0.988 | 308 | 0 | 0 |
| thrash | handcrafted | 1.000 | 0.985 | 0.985 | 261 | 0 | 0 |
| runaway_cost | mamba | 1.000 | 0.984 | 0.984 | 229 | 0 | 0 |

## Macro-F1 Across Detectors

Mean F1 across the five detectors at each cutoff. Useful as a single-number paradigm summary, but the per-detector tables above are the authoritative source for paper claims.

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 0.466 | 0.885 | 0.946 | 0.967 |
| causal | 0.000 | 0.218 | 0.373 | 0.379 |
| tda | 0.000 | 0.000 | 0.531 | 0.912 |
| mamba | 0.000 | 0.642 | 0.972 | 0.998 |
| hopfield | 0.901 | 0.961 | 0.903 | 0.992 |

## Coverage by Cutoff

Number of traces evaluated per (paradigm, cutoff) cell. The harness uses `skip_short_traces=False`, so traces shorter than a cutoff are passed through at their full length rather than skipped — meaning every paradigm sees the full filtered corpus at every cutoff. This table is therefore informational; eligibility gaps surface in the F1 / recall tables above as drops on traces the paradigm can't process.

| Paradigm | 3 steps | 5 steps | 7 steps | Full |
|---|---|---|---|---|
| handcrafted | 1494/1494 | 1494/1494 | 1494/1494 | 1494/1494 |
| causal | 1494/1494 | 1494/1494 | 1494/1494 | 1494/1494 |
| tda | 1494/1494 | 1494/1494 | 1494/1494 | 1494/1494 |
| mamba | 1494/1494 | 1494/1494 | 1494/1494 | 1494/1494 |
| hopfield | 1494/1494 | 1494/1494 | 1494/1494 | 1494/1494 |

## Failure Mode Analysis

For each detector, identifies which paradigm holds the lead at each cutoff. Where the leader changes between cutoffs, that's a signal that different paradigms shine at different stages of trace progression — discussion fodder for the eventual paper.

### loop
- **3 steps**: hopfield (F1 0.936, P_lb 0.833, R_lb 0.982, TP=212 FP=29 FN=0)
- **5 steps**: hopfield (F1 0.986, P_lb 0.982, R_lb 0.940, TP=206 FP=0 FN=6)
- **7 steps**: hopfield (F1 0.960, P_lb 0.927, R_lb 0.921, TP=203 FP=8 FN=9)
- **Full**: handcrafted (F1 1.000, P_lb 0.982, R_lb 0.982, TP=212 FP=0 FN=0)
- **Cutoff-dependent leader**: handcrafted / hopfield — different paradigms hold the lead at different stages.

### stuck
- **3 steps**: hopfield (F1 0.890, P_lb 0.736, R_lb 0.973, TP=138 FP=34 FN=0)
- **5 steps**: hopfield (F1 0.899, P_lb 0.802, R_lb 0.881, TP=129 FP=20 FN=9)
- **7 steps**: mamba (F1 0.989, P_lb 0.939, R_lb 0.973, TP=138 FP=3 FN=0)
- **Full**: mamba (F1 0.989, P_lb 0.939, R_lb 0.973, TP=138 FP=3 FN=0)
- **Cutoff-dependent leader**: hopfield / mamba — different paradigms hold the lead at different stages.

### confabulation
- **3 steps**: hopfield (F1 0.901, P_lb 0.798, R_lb 0.945, TP=299 FP=57 FN=9)
- **5 steps**: hopfield (F1 0.920, P_lb 0.850, R_lb 0.925, TP=294 FP=37 FN=14)
- **7 steps**: mamba (F1 0.989, P_lb 0.955, R_lb 0.988, TP=308 FP=7 FN=0)
- **Full**: mamba (F1 1.000, P_lb 0.988, R_lb 0.988, TP=308 FP=0 FN=0)
- **Cutoff-dependent leader**: hopfield / mamba — different paradigms hold the lead at different stages.

### thrash
- **3 steps**: hopfield (F1 0.941, P_lb 0.847, R_lb 0.985, TP=261 FP=33 FN=0)
- **5 steps**: handcrafted (F1 1.000, P_lb 0.985, R_lb 0.985, TP=261 FP=0 FN=0)
- **7 steps**: handcrafted (F1 1.000, P_lb 0.985, R_lb 0.985, TP=261 FP=0 FN=0)
- **Full**: handcrafted (F1 1.000, P_lb 0.985, R_lb 0.985, TP=261 FP=0 FN=0)
- **Cutoff-dependent leader**: handcrafted / hopfield — different paradigms hold the lead at different stages.

### runaway_cost
- **3 steps**: hopfield (F1 0.836, P_lb 0.666, R_lb 0.984, TP=229 FP=90 FN=0)
- **5 steps**: hopfield (F1 1.000, P_lb 0.984, R_lb 0.984, TP=229 FP=0 FN=0)
- **7 steps**: mamba (F1 1.000, P_lb 0.984, R_lb 0.984, TP=229 FP=0 FN=0)
- **Full**: mamba (F1 1.000, P_lb 0.984, R_lb 0.984, TP=229 FP=0 FN=0)
- **Cutoff-dependent leader**: hopfield / mamba — different paradigms hold the lead at different stages.

## Production Deployment Guidance

Per-detector recommendations derived programmatically from the per-detector winner tables. These are bench data; final integration choices in agent-vitals should weigh them against compute cost, dependency footprint, and the production framing already in v1.13.x (handcrafted rules + causal + TDA hybrid override-only).

| Detector | Production Pick | Why |
|---|---|---|
| loop | handcrafted (+hopfield early) | handcrafted leads full-trace (F1 1.000); consider hopfield as an early-detection adjudicator (F1 @3 0.936) |
| stuck | mamba (+hopfield early) | mamba leads full-trace (F1 0.989); consider hopfield as an early-detection adjudicator (F1 @3 0.890) |
| confabulation | mamba (+hopfield early) | mamba leads full-trace (F1 1.000); consider hopfield as an early-detection adjudicator (F1 @3 0.901) |
| thrash | handcrafted (+hopfield early) | handcrafted leads full-trace (F1 1.000); consider hopfield as an early-detection adjudicator (F1 @3 0.941) |
| runaway_cost | mamba (+hopfield early) | mamba leads full-trace (F1 1.000); consider hopfield as an early-detection adjudicator (F1 @3 0.836) |

## Reproducibility

Run the report end-to-end from the sibling tda-experiment Python 3.12 venv (which has torch, gtda, hflayers, agent-vitals editable):

```
make five-paradigm-comparative
```

Or directly:

```
PYTHONPATH=$(pwd) ../tda-experiment/.venv/bin/python scripts/run_five_paradigm_comparative.py
```

Every figure in this report is regenerable from `reports/eval-five-paradigm-comparative.json` (paradigm × cutoff × detector cell list with TP/FP/FN/TN, P/R/F1, P_lb/R_lb, and per-cell evaluated trace count) without re-running the detectors.

### Per-paradigm runtime

| Paradigm | Total seconds |
|---|---|
| handcrafted | 4.6 |
| causal | 0.2 |
| tda | 19.3 |
| mamba | 75.9 |
| hopfield | 19.0 |

