"""Prototype detector modules used for bench-side validation."""

from importlib import import_module

__all__ = [
    "CausalConfabConfig",
    "CausalConfabEvaluation",
    "CausalConfabResult",
    "CausalRunawayConfig",
    "CausalRunawayEvaluation",
    "CausalRunawayResult",
    "TDAConfig",
    "TDAEvaluationMetric",
    "TDAPrediction",
    "TDATrainingSummary",
    "WindowScore",
    "RunawayWindowScore",
    "MultiplierSweepResult",
    "predict_detectors",
    "train_tda_models",
    "detect_causal_confabulation",
    "detect_causal_runaway_cost",
    "evaluate_causal_confab_corpus",
    "evaluate_causal_runaway_corpus",
    "score_causal_windows",
    "score_runaway_windows",
    "sweep_cost_growth_multiplier",
]


def __getattr__(name: str) -> object:
    if name in {
        "TDAConfig",
        "TDAEvaluationMetric",
        "TDAPrediction",
        "TDATrainingSummary",
        "predict_detectors",
        "train_tda_models",
    }:
        module = import_module("prototypes.tda_detector")
        return getattr(module, name)
    if name in __all__:
        module = import_module("prototypes.causal_confab")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
