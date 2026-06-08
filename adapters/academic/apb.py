"""APB -> VitalsSnapshot adapter (bfcl / gaia_dev / hotpotqa / tau2).

All 4 APB families share the same raw-record schema (question, messages,
step_labels, final_label, data_source, ground_truth). Only the workflow-
type tag differs per family (bfcl/tau2 are build-style, gaia_dev/hotpotqa
are research-style — see agents.md resolve_workflow_type conventions).

The adapter preserves APB's step-indexing convention: step_labels keys are
raw message indices pointing at assistant turns, and the trace's any-
negative-step binary label (`y` in the metrics TR pipeline) is derived
exactly as the upstream TR-20260418-001 adapter does it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from agent_vitals.schema import VitalsSnapshot

from adapters.academic._base import build_snapshots_from_chat

APBFamily = Literal["bfcl", "gaia_dev", "hotpotqa", "tau2"]

APB_FAMILIES: tuple[APBFamily, ...] = ("bfcl", "gaia_dev", "hotpotqa", "tau2")

# agents.md resolve_workflow_type: `.bc.` -> build, `.rc.` -> research.
# bfcl is function-calling in a gorilla file system (build),
# tau2 is a multi-turn customer/retail agent (build),
# gaia_dev and hotpotqa are research-QA workflows.
_FAMILY_WORKFLOW_TAG: dict[APBFamily, str] = {
    "bfcl": "bc",
    "tau2": "bc",
    "gaia_dev": "rc",
    "hotpotqa": "rc",
}


@dataclass(frozen=True)
class APBTrace:
    """Structured APB record with a derived binary incident label."""

    record: dict[str, Any]
    family: APBFamily
    trace_id: str
    incident: bool

    @property
    def final_label_raw(self) -> int:
        return int(self.record.get("final_label", 0))


def build_trace_id(record: dict[str, Any], family: APBFamily) -> str:
    """Build a bench-convention trace_id.

    Embeds `.bc.` or `.rc.` so evaluator.runner.resolve_workflow_type()
    picks the correct workflow type automatically when the adapter output
    is later loaded via the standard manifest path.
    """
    query_index = record.get("query_index")
    sample_index = record.get("sample_index")
    tag = _FAMILY_WORKFLOW_TAG[family]
    return f"apb.{family}.{tag}.{query_index}.{sample_index}"


def _incident_from_step_labels(record: dict[str, Any]) -> bool:
    step_labels = record.get("step_labels") or {}
    return any(int(v) < 0 for v in step_labels.values())


def apb_to_vitals_snapshots(
    record: dict[str, Any],
    family: APBFamily,
) -> list[VitalsSnapshot]:
    """Convert a single APB record into a VitalsSnapshot stream."""
    if family not in APB_FAMILIES:
        raise ValueError(f"Unknown APB family: {family!r}")
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        raise TypeError(f"APB messages field must be a list, got {type(messages).__name__}")
    trace_id = build_trace_id(record, family)
    return build_snapshots_from_chat(
        messages,
        mission_id=f"apb-{family}-{record.get('query_index')}-{record.get('sample_index')}",
        run_id=trace_id,
    )


def iter_apb_traces(
    corpus_path: Path,
    family: APBFamily,
) -> Iterator[APBTrace]:
    """Iterate the raw records for one APB family with derived labels.

    `corpus_path` should be the corpus/academic/apb directory; records are
    read from `corpus_path / f"{family}.jsonl"`.
    """
    if family not in APB_FAMILIES:
        raise ValueError(f"Unknown APB family: {family!r}")
    jsonl = corpus_path / f"{family}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"APB family file missing: {jsonl}")
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            yield APBTrace(
                record=record,
                family=family,
                trace_id=build_trace_id(record, family),
                incident=_incident_from_step_labels(record),
            )
