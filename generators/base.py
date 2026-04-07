"""Base types and helpers for synthetic trace generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
    framework: str | None = None

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
                "framework": self.framework,
                "reviewer": None,
                "review_date": None,
                "confidence": self.confidence,
                "notes": self.notes,
            },
        }


# Per-framework signal adjustments derived from agent-vitals thresholds.yaml profiles.
# Each framework's thresholds differ, so generators must calibrate signals accordingly.
FRAMEWORK_PROFILES: dict[str, dict[str, Any]] = {
    "langgraph": {
        # loop_consecutive_pct=0.4 (lower → loop triggers earlier)
        # burn_rate_multiplier=2.5 (lower → runaway triggers at less burn)
        "loop_consecutive_pct": 0.4,
        "burn_rate_multiplier": 2.5,
    },
    "crewai": {
        # burn_rate_multiplier=3.0 (higher → needs more burn for runaway)
        # token_scale_factor=0.7 (tokens are scaled down)
        "burn_rate_multiplier": 3.0,
        "token_scale_factor": 0.7,
    },
    "dspy": {
        # loop_consecutive_pct=0.7 (higher → needs longer plateau for loop)
        # stuck_dm_threshold=0.1 (lower → stuck triggers at lower dm)
        # workflow_stuck_enabled=none (stuck is disabled)
        "loop_consecutive_pct": 0.7,
        "stuck_dm_threshold": 0.1,
        "workflow_stuck_enabled": "none",
    },
}


class TraceGenerator:
    """Shared helpers for synthetic trace generators."""

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
