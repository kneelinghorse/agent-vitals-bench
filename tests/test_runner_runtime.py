"""Replay/runtime tests for the evaluator runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_vitals import VitalsConfig
from agent_vitals.backtest import _replay_trace as production_replay_trace

from evaluator.runner import (
    DETECTORS,
    load_manifest,
    load_trace,
    replay_trace,
    resolve_workflow_type,
)


def test_replay_trace_matches_production_default_replay() -> None:
    """Bench replay must stay bit-aligned with production detector semantics."""
    cfg = VitalsConfig.from_yaml(allow_env_override=False)
    manifest = load_manifest("v1")
    mismatches: list[str] = []

    for entry in manifest:
        confidence = float(entry.get("metadata", {}).get("confidence", 0.0) or 0.0)
        if confidence < 0.8:
            continue

        snapshots = load_trace("v1", str(entry["path"]))
        mission_id = snapshots[0].mission_id if snapshots else None
        workflow_type = resolve_workflow_type(str(entry["trace_id"]), mission_id)

        expected = production_replay_trace(
            snapshots,
            config=cfg,
            workflow_type=workflow_type,
        )
        actual = replay_trace(
            snapshots,
            config=cfg,
            workflow_type=workflow_type,
            runtime_mode="default",
        )

        expected_normalized = {detector: bool(expected[detector]) for detector in DETECTORS}
        if actual != expected_normalized:
            mismatches.append(str(entry["trace_id"]))

    assert not mismatches, f"bench replay drifted on {len(mismatches)} traces: {mismatches[:10]}"


def test_replay_trace_runtime_mode_controls_tda_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit runtime modes should deterministically control TDA enablement."""
    observed_tda_flags: list[bool] = []

    def fake_replay_trace(*_args: object, **kwargs: object) -> dict[str, bool]:
        config = kwargs["config"]
        observed_tda_flags.append(bool(config.tda_enabled))
        return {
            "loop": False,
            "stuck": False,
            "confabulation": False,
            "thrash": False,
            "runaway_cost": False,
            "any": False,
        }

    monkeypatch.setattr("evaluator.runner._production_replay_trace", fake_replay_trace)

    manifest = load_manifest("v1")
    sample_path = Path(str(manifest[0]["path"]))
    snapshots = load_trace("v1", str(sample_path))
    cfg = VitalsConfig.from_yaml(allow_env_override=False)

    replay_trace(snapshots, config=cfg, workflow_type="unknown", runtime_mode="default")
    replay_trace(snapshots, config=cfg, workflow_type="unknown", runtime_mode="tda")

    assert observed_tda_flags == [False, True]
