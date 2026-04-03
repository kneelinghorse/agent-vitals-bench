"""Synthetic trace generators for Agent Vitals Bench."""

from generators.base import TraceGenerator, TraceMetadata
from generators.loop import LoopGenerator
from generators.stuck import StuckGenerator
from generators.confabulation import ConfabGenerator
from generators.thrash import ThrashGenerator
from generators.runaway_cost import RunawayCostGenerator

__all__ = [
    "TraceGenerator",
    "TraceMetadata",
    "LoopGenerator",
    "StuckGenerator",
    "ConfabGenerator",
    "ThrashGenerator",
    "RunawayCostGenerator",
]
