"""Golden-trace tests for SWE-verified / CoderForge -> VitalsSnapshot adapter."""

from __future__ import annotations

from pathlib import Path

from agent_vitals.schema import VitalsSnapshot

from adapters.academic.swe_verified_coderforge import (
    iter_swe_traces,
    swe_to_vitals_snapshots,
)
from evaluator.runner import DETECTORS, replay_trace

CORPUS = Path(__file__).resolve().parents[3] / "corpus" / "academic" / "swe_verified_coderforge"

_GOLDEN_COUNT = 5


def test_swe_trajectory_file_present() -> None:
    assert (CORPUS / "trajectories.jsonl").exists()


def test_swe_adapter_produces_valid_snapshots() -> None:
    traces = list(iter_swe_traces(CORPUS))
    assert len(traces) >= _GOLDEN_COUNT
    for trace in traces[:_GOLDEN_COUNT]:
        snapshots = swe_to_vitals_snapshots(trace.record)
        assert snapshots, f"empty snapshots for {trace.trace_id}"
        assert all(isinstance(s, VitalsSnapshot) for s in snapshots)
        assert all(s.loop_index == i for i, s in enumerate(snapshots))


def test_swe_replay_trace_returns_full_detector_set() -> None:
    traces = list(iter_swe_traces(CORPUS))[:_GOLDEN_COUNT]
    for trace in traces:
        snapshots = swe_to_vitals_snapshots(trace.record)
        predictions = replay_trace(snapshots, workflow_type="build")
        assert set(predictions.keys()) == set(DETECTORS)
        assert all(isinstance(v, bool) for v in predictions.values())


def test_trace_id_embeds_build_tag() -> None:
    first = next(iter_swe_traces(CORPUS))
    assert ".bc." in first.trace_id


def test_reward_binarization_matches_metrics_convention() -> None:
    """reward < 1.0 -> incident=1, per TR-20260418-001 convention."""
    for trace in list(iter_swe_traces(CORPUS))[:20]:
        if trace.reward < 1.0:
            assert trace.incident is True
        else:
            assert trace.incident is False
