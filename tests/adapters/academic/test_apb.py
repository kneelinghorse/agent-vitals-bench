"""Golden-trace tests for APB -> VitalsSnapshot adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_vitals.schema import VitalsSnapshot

from adapters.academic.apb import (
    APB_FAMILIES,
    APBFamily,
    apb_to_vitals_snapshots,
    iter_apb_traces,
)
from evaluator.runner import DETECTORS, replay_trace

CORPUS = Path(__file__).resolve().parents[3] / "corpus" / "academic" / "apb"

_GOLDEN_COUNT = 5


@pytest.mark.parametrize("family", APB_FAMILIES)
def test_apb_family_file_present(family: APBFamily) -> None:
    assert (CORPUS / f"{family}.jsonl").exists(), f"{family}.jsonl missing"


@pytest.mark.parametrize("family", APB_FAMILIES)
def test_apb_adapter_produces_valid_snapshots(family: APBFamily) -> None:
    traces = list(iter_apb_traces(CORPUS, family))
    assert len(traces) >= _GOLDEN_COUNT, f"{family} has fewer than {_GOLDEN_COUNT} traces"
    for trace in traces[:_GOLDEN_COUNT]:
        snapshots = apb_to_vitals_snapshots(trace.record, family)
        assert snapshots, f"empty snapshots for {trace.trace_id}"
        assert all(isinstance(s, VitalsSnapshot) for s in snapshots)
        assert all(s.loop_index == i for i, s in enumerate(snapshots))


@pytest.mark.parametrize("family", APB_FAMILIES)
def test_apb_replay_trace_returns_full_detector_set(family: APBFamily) -> None:
    traces = list(iter_apb_traces(CORPUS, family))[:_GOLDEN_COUNT]
    workflow = "build" if family in ("bfcl", "tau2") else "research"
    for trace in traces:
        snapshots = apb_to_vitals_snapshots(trace.record, family)
        predictions = replay_trace(snapshots, workflow_type=workflow)
        assert set(predictions.keys()) == set(DETECTORS)
        assert all(isinstance(v, bool) for v in predictions.values())


def test_trace_id_embeds_workflow_tag() -> None:
    bfcl = next(iter_apb_traces(CORPUS, "bfcl"))
    hotpot = next(iter_apb_traces(CORPUS, "hotpotqa"))
    assert ".bc." in bfcl.trace_id
    assert ".rc." in hotpot.trace_id
