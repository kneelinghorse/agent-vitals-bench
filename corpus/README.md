# Agent Vitals Bench — Corpus

Labeled trace corpus for evaluating Agent Vitals detectors.

## Structure

```
corpus/
├── legacy/           AV-34 traces (re-reviewed)
│   ├── manifest.json
│   └── traces/
└── v1/               Bench corpus (synthetic + elicited)
    ├── manifest.json
    └── traces/
        ├── loop/{positive,negative}/
        ├── stuck/{positive,negative}/
        ├── confabulation/{positive,negative}/
        ├── thrash/{positive,negative}/
        └── runaway_cost/{positive,negative}/
```

## File Format

Traces are JSON arrays of `VitalsSnapshot` objects. See `docs/corpus_schema.md` for full schema.

## Naming Convention

`{detector}-{tier}-{sequence}.json`
- `loop-syn-001.json` — loop, synthetic
- `confab-eli-012.json` — confabulation, elicited
- `stuck-leg-003.json` — stuck, legacy

## Gate Targets

Per detector: 40+ positives, 40+ negatives. Wilson CI 95%: P_lb >= 0.80, R_lb >= 0.75.
