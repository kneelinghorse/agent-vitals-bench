"""SWE-verified / CoderForge -> VitalsSnapshot adapter.

CoderForge trajectories ship their chat transcripts as a JSON-encoded
string inside the `messages` field (not as a native list). We parse it
lazily per record and feed the resulting message list into the shared
chat-to-snapshot derivation.

Label: `reward < 1.0` -> incident=1, per metrics TR-20260418-001 convention
(failure is the positive class for detection). SWE trajectories are build-
workflow, so the trace_id carries a `.bc.` tag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agent_vitals.schema import VitalsSnapshot

from adapters.academic._base import build_snapshots_from_chat


@dataclass(frozen=True)
class SWETrace:
    """Structured SWE+CoderForge record with a derived binary incident label."""

    record: dict[str, Any]
    trace_id: str
    incident: bool

    @property
    def reward(self) -> float:
        return float(self.record.get("reward") or 0.0)


def _parse_messages(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        raise TypeError(
            f"CoderForge messages JSON must decode to a list, got {type(parsed).__name__}"
        )
    raise TypeError(
        f"CoderForge messages field must be list or JSON string, got {type(raw).__name__}"
    )


def build_trace_id(record: dict[str, Any]) -> str:
    """SWE trace_id carries a `.bc.` tag so bench resolves workflow_type=build."""
    base = record.get("trajectory_id") or (
        f"{record.get('instance_id')}:run{record.get('run_id', 1)}"
    )
    return f"swe.bc.{base}"


def swe_to_vitals_snapshots(record: dict[str, Any]) -> list[VitalsSnapshot]:
    """Convert a single SWE+CoderForge record into a VitalsSnapshot stream."""
    messages = _parse_messages(record.get("messages"))
    trace_id = build_trace_id(record)
    return build_snapshots_from_chat(
        messages,
        mission_id=f"swe-{record.get('instance_id')}",
        run_id=trace_id,
    )


def iter_swe_traces(corpus_path: Path) -> Iterator[SWETrace]:
    """Iterate SWE+CoderForge trajectories with derived labels.

    `corpus_path` should be the corpus/academic/swe_verified_coderforge
    directory; records are read from `corpus_path / 'trajectories.jsonl'`.
    """
    jsonl = corpus_path / "trajectories.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"SWE trajectory file missing: {jsonl}")
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            reward = float(record.get("reward") or 0.0)
            incident = reward < 1.0
            yield SWETrace(
                record=record,
                trace_id=build_trace_id(record),
                incident=incident,
            )
