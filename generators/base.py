"""Base class and types for synthetic trace generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_vitals.schema import VitalsSnapshot


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    """Ground-truth metadata for a generated trace."""

    trace_id: str
    generator: str
    tier: str = "synthetic"
    labels: dict[str, bool] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    onset_step: int | None = None
    confidence: float = 1.0
    notes: str = ""

    def as_manifest_entry(self, path: str) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "path": path,
            "tier": self.tier,
            "labels": self.labels,
            "metadata": {
                "generator": self.generator,
                "params": self.params,
                "onset_step": self.onset_step,
                "tier": self.tier,
                "model": None,
                "provider": None,
                "framework": None,
                "reviewer": None,
                "review_date": None,
                "confidence": self.confidence,
                "notes": self.notes,
            },
        }


class TraceGenerator(ABC):
    """Base class for synthetic trace generators."""

    @abstractmethod
    def generate(self, **kwargs: Any) -> tuple[list[VitalsSnapshot], TraceMetadata]:
        """Generate a trace with known ground truth.

        Returns:
            Tuple of (snapshots, metadata) where metadata contains labels
            and generator parameters for the manifest.
        """
        ...

    @staticmethod
    def _make_timestamp(step: int) -> datetime:
        """Deterministic timestamp for reproducibility."""
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _default_labels() -> dict[str, bool]:
        return {
            "loop": False,
            "stuck": False,
            "confabulation": False,
            "thrash": False,
            "runaway_cost": False,
        }
