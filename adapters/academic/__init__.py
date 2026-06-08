"""Academic benchmark -> VitalsSnapshot adapters (Sprint 22 M02).

Public entry points:
    - apb.apb_to_vitals_snapshots(record, family)
    - apb.iter_apb_traces(corpus_path, family)
    - swe_verified_coderforge.swe_to_vitals_snapshots(record)
    - swe_verified_coderforge.iter_swe_traces(corpus_path)

See docs/academic-adapter-gaps.md for the per-family signal-derivation
contract and known coverage gaps.
"""
