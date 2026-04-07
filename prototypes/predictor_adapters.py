"""TracePredictor adapters for the five detection paradigms.

Each ``make_*_predictor`` factory returns a callable conforming to
``evaluator.partial_trace.TracePredictor``: it accepts a
``Sequence[VitalsSnapshot]`` and returns a ``Mapping[str, bool]`` keyed by the
canonical bench detector names (loop / stuck / confabulation / thrash /
runaway_cost).

These adapters exist so a single harness invocation
(``evaluate_partial_traces``) can score every paradigm through identical
plumbing for the Five-Paradigm Comparative Report (sprint 16, mission s16-m01).

Conventions
-----------
- **Standalone semantics**, not hybrid: each paradigm's predictions stand on
  their own. Adapters do not silently fall back to handcrafted when the
  paradigm cannot answer; instead they return ``False`` for the unanswerable
  detectors. The harness reports per-cell trace counts so eligibility gaps
  are visible.
- **Lazy backend loading**: torch / sklearn / gtda imports are deferred to the
  first call. Importing this module from the bench main venv (no ML deps) is
  safe; only invoking the learned-paradigm factories triggers the heavy
  imports.
- **Per-cutoff Hopfield**: Hopfield ships separate models for prefix lengths
  3 / 5 / 7 plus a full-trace model. ``make_hopfield_predictor`` accepts a
  ``prefix_len`` argument so the comparative harness can pick the matching
  model for each cutoff being evaluated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_vitals.schema import VitalsSnapshot

from evaluator.partial_trace import (
    TracePredictor,
    make_handcrafted_predictor,
)
from evaluator.runner import DETECTORS

PARADIGMS: tuple[str, ...] = ("handcrafted", "causal", "tda", "mamba", "hopfield")


__all__ = [
    "DETECTORS",
    "PARADIGMS",
    "TracePredictor",
    "make_causal_predictor",
    "make_handcrafted_predictor",
    "make_hopfield_predictor",
    "make_mamba_predictor",
    "make_tda_predictor",
]


def _all_false() -> dict[str, bool]:
    return {detector: False for detector in DETECTORS}


def _snapshot_to_dict(snapshot: VitalsSnapshot) -> dict[str, Any]:
    """Convert a VitalsSnapshot pydantic model to the dict shape that the
    Mamba and Hopfield prototype detectors consume.

    The learned detectors call ``snapshot.get("signals", {})`` style accessors
    rather than attribute access, so a Pydantic ``model_dump()`` round-trip is
    required.
    """
    return snapshot.model_dump()


# ---------------------------------------------------------------------------
# Causal paradigm
# ---------------------------------------------------------------------------


def make_causal_predictor(
    *,
    confab_config: Any | None = None,
    runaway_config: Any | None = None,
) -> TracePredictor:
    """Build a TracePredictor backed by the causal prototype detectors.

    The causal prototypes only cover ``confabulation`` and ``runaway_cost``;
    the remaining three detectors return ``False`` (standalone semantics, no
    handcrafted fallback). When the trace is too short for the rolling causal
    windows the detectors return ``detected=False`` with an
    ``insufficient_history`` trigger, which the harness scores as a negative.
    """
    from prototypes.causal_confab import detect_causal_confabulation
    from prototypes.causal_runaway import detect_causal_runaway_cost

    def predictor(snapshots: Sequence[VitalsSnapshot]) -> dict[str, bool]:
        result = _all_false()
        if not snapshots:
            return result
        confab = detect_causal_confabulation(snapshots, confab_config)
        runaway = detect_causal_runaway_cost(snapshots, runaway_config)
        result["confabulation"] = bool(confab.detected)
        result["runaway_cost"] = bool(runaway.detected)
        return result

    return predictor


# ---------------------------------------------------------------------------
# TDA paradigm
# ---------------------------------------------------------------------------


def make_tda_predictor(
    *,
    model_dir: Path | None = None,
    detectors: Sequence[str] = DETECTORS,
) -> TracePredictor:
    """Build a TracePredictor backed by the bench TDA gradient-boosting models.

    The bench TDA prototype trains one ``GradientBoostingClassifier`` per
    detector on persistence-diagram features and saves them under
    ``prototypes/tda_models/{detector}.joblib``. The ``predict_detectors``
    helper accepts ``Sequence[VitalsSnapshot]`` directly because its internal
    ``_snapshot_section`` handles both Mapping and Pydantic inputs.

    Returns all-False for the detectors when ``extract_tda_features`` cannot
    produce a vector (trace shorter than ``max(window_sizes) + 2`` steps,
    default 7). This is the correct standalone behavior — TDA simply has no
    opinion on traces below its window floor.
    """
    from prototypes.tda_detector import (
        TDAConfig,
        DEFAULT_MODEL_DIR as TDA_DEFAULT_MODEL_DIR,
        load_predictor as load_tda_predictor,
    )

    resolved_dir = model_dir if model_dir is not None else TDA_DEFAULT_MODEL_DIR
    cached_predict = load_tda_predictor(
        detectors=tuple(detectors),
        model_dir=resolved_dir,
        config=TDAConfig(),
    )

    def predictor(snapshots: Sequence[VitalsSnapshot]) -> dict[str, bool]:
        result = _all_false()
        if not snapshots:
            return result
        try:
            preds = cached_predict(list(snapshots))
        except ValueError:
            # Trace too short for feature extraction.
            return result
        for detector, prediction in preds.items():
            result[detector] = bool(prediction.detected)
        return result

    return predictor


# ---------------------------------------------------------------------------
# Mamba paradigm
# ---------------------------------------------------------------------------


def make_mamba_predictor(
    *,
    model_dir: Path | None = None,
    min_steps: int = 5,
) -> TracePredictor:
    """Build a TracePredictor backed by the bench Mamba multilabel model.

    Loads ``prototypes/mamba_models/multilabel.pt`` lazily on first call and
    caches the model artifact for the lifetime of the predictor. The model
    is multi-label and returns predictions for all 5 detectors in a single
    forward pass.

    Mamba expects step dictionaries (not Pydantic objects), so each snapshot
    is converted via ``model_dump()`` before being passed to
    ``predict_multilabel``. Traces shorter than ``min_steps`` get an all-False
    response — the model was trained with ``min_steps=5`` and is not expected
    to give meaningful predictions on shorter prefixes.
    """
    from prototypes.mamba_detector import (
        DEFAULT_MODEL_DIR as MAMBA_DEFAULT_MODEL_DIR,
        load_multilabel_predictor,
    )

    resolved_dir = model_dir if model_dir is not None else MAMBA_DEFAULT_MODEL_DIR
    cached_predict = load_multilabel_predictor(model_dir=resolved_dir)

    def predictor(snapshots: Sequence[VitalsSnapshot]) -> dict[str, bool]:
        result = _all_false()
        if not snapshots or len(snapshots) < min_steps:
            return result
        steps = [_snapshot_to_dict(snap) for snap in snapshots]
        preds = cached_predict(steps)
        for detector, prediction in preds.items():
            if detector in result:
                result[detector] = bool(prediction.detected)
        return result

    return predictor


# ---------------------------------------------------------------------------
# Hopfield paradigm
# ---------------------------------------------------------------------------


HOPFIELD_PREFIX_FOR_CUTOFF: dict[int | None, int | None] = {
    3: 3,
    5: 5,
    7: 7,
    None: None,
}


def make_hopfield_predictor(
    *,
    prefix_len: int | None = None,
    model_dir: Path | None = None,
    min_steps: int | None = None,
    detectors: Sequence[str] = DETECTORS,
) -> TracePredictor:
    """Build a TracePredictor backed by the bench Hopfield prefix models.

    Hopfield ships per-cutoff binary classifiers under
    ``prototypes/hopfield_models/{detector}_p{prefix_len}.pt``. Pass
    ``prefix_len=3 / 5 / 7`` to load the matching prefix-trained models, or
    ``prefix_len=None`` (default) to load the full-trace models.

    ``min_steps`` defaults to ``prefix_len`` when a prefix model is selected
    (since the prefix models were trained specifically on that prefix length
    and should accept inputs of exactly that length), and to 5 for the
    full-trace model (matching ``HopfieldConfig.min_steps``).

    Each detector is a separate binary model. The artifacts are loaded once
    on first call and cached on the closure for the lifetime of the
    predictor.
    """
    from prototypes.hopfield_detector import (
        DEFAULT_MODEL_DIR as HOPFIELD_DEFAULT_MODEL_DIR,
        load_predictor as load_hopfield_predictor,
    )

    resolved_dir = model_dir if model_dir is not None else HOPFIELD_DEFAULT_MODEL_DIR
    if min_steps is None:
        min_steps = prefix_len if prefix_len is not None else 5
    effective_min_steps = min_steps

    cached_predictors: dict[str, Any] = {}
    for detector in detectors:
        try:
            cached_predictors[detector] = load_hopfield_predictor(
                detector,
                prefix_len=prefix_len,
                model_dir=resolved_dir,
                include_energy=False,
            )
        except FileNotFoundError:
            continue

    def predictor(snapshots: Sequence[VitalsSnapshot]) -> dict[str, bool]:
        result = _all_false()
        if not snapshots or len(snapshots) < effective_min_steps:
            return result
        steps = [_snapshot_to_dict(snap) for snap in snapshots]
        for detector, cached in cached_predictors.items():
            prediction = cached(steps)
            if detector in result:
                result[detector] = bool(prediction.detected)
        return result

    return predictor


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def build_predictor(
    paradigm: str,
    *,
    cutoff: int | None = None,
    model_dirs: Mapping[str, Path] | None = None,
) -> TracePredictor:
    """Construct a paradigm predictor by name.

    ``cutoff`` is only consulted for the Hopfield paradigm to pick the matching
    prefix-trained model. The other paradigms ignore it.

    ``model_dirs`` may map paradigm names to model directories to override the
    defaults under ``prototypes/{paradigm}_models``.
    """
    if paradigm not in PARADIGMS:
        raise ValueError(f"unknown paradigm: {paradigm!r} (expected one of {PARADIGMS})")

    overrides = dict(model_dirs or {})

    if paradigm == "handcrafted":
        return make_handcrafted_predictor()
    if paradigm == "causal":
        return make_causal_predictor()
    if paradigm == "tda":
        return make_tda_predictor(model_dir=overrides.get("tda"))
    if paradigm == "mamba":
        return make_mamba_predictor(model_dir=overrides.get("mamba"))
    if paradigm == "hopfield":
        prefix_len = HOPFIELD_PREFIX_FOR_CUTOFF.get(cutoff, None)
        return make_hopfield_predictor(
            prefix_len=prefix_len,
            model_dir=overrides.get("hopfield"),
        )
    raise AssertionError(f"unreachable paradigm dispatch: {paradigm}")  # pragma: no cover
