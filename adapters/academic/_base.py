"""Shared chat-transcript -> VitalsSnapshot derivation for academic corpora.

These corpora (APB, SWE-verified/CoderForge) ship only chat transcripts +
run-level labels. They do not carry the telemetry fields that agent-vitals
normally consumes (findings/sources/coverage/tokens/errors/CUSUM). This
module derives the smallest plausible set of RawSignals from message
structure so replay_trace() can run through every trace without errors.

Per-family calibration and label mapping live in the family-specific
adapters; this module is signal-agnostic beyond message-structure features.
See docs/academic-adapter-gaps.md for the full derivation contract.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from agent_vitals.detection.metrics import TemporalMetrics
from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot


_ERROR_PATTERNS = re.compile(
    r"\b(error|exception|traceback|failed|failure|invalid|cannot|could not)\b",
    flags=re.IGNORECASE,
)
_METRICS = TemporalMetrics()


@dataclass(frozen=True)
class ChatStep:
    """One assistant turn plus any tool responses that follow it.

    The step is our canonical unit of evaluation: each assistant emission
    (text reply and/or tool_calls) produces one VitalsSnapshot. Tool results
    consumed by the assistant are attributed to the assistant step that
    issued them, since that is what APB's step_labels key against.
    """

    message_index: int
    assistant: dict[str, Any]
    tool_responses: list[dict[str, Any]]

    @property
    def text(self) -> str:
        content = self.assistant.get("content") or ""
        return content if isinstance(content, str) else str(content)

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for call in self.assistant.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(str(name))
        return names

    @property
    def tool_response_text(self) -> str:
        return "\n".join(str(m.get("content") or "") for m in self.tool_responses)


def segment_into_steps(messages: list[dict[str, Any]]) -> list[ChatStep]:
    """Walk a chat transcript and group it into assistant-centric steps.

    An assistant message starts a new step; any immediately-following tool
    messages are attached to it. User and system messages do not create
    steps (they precede / interleave with assistant turns).
    """
    steps: list[ChatStep] = []
    current: dict[str, Any] | None = None
    current_index: int = -1
    current_tools: list[dict[str, Any]] = []
    for index, msg in enumerate(messages):
        role = (msg or {}).get("role")
        if role == "assistant":
            if current is not None:
                steps.append(ChatStep(current_index, current, current_tools))
            current = msg
            current_index = index
            current_tools = []
        elif role == "tool" and current is not None:
            current_tools.append(msg)
    if current is not None:
        steps.append(ChatStep(current_index, current, current_tools))
    return steps


def _char_to_token_estimate(text: str) -> int:
    """Crude character->token estimate (~4 chars/token, OpenAI tokenizer order-of-magnitude)."""
    return max(0, len(text) // 4)


def _build_raw_signals(
    step: ChatStep,
    *,
    cumulative_findings: int,
    cumulative_tokens: int,
    cumulative_errors: int,
    unique_tool_names: set[str],
    expected_findings: int,
) -> RawSignals:
    step_text = step.text
    step_tool_text = step.tool_response_text
    step_tool_names = set(step.tool_names)
    unique_tool_names_so_far = unique_tool_names | step_tool_names

    produced_output = bool(step_text.strip() or step.tool_names)
    findings = cumulative_findings + (1 if produced_output else 0)

    step_text_tokens = _char_to_token_estimate(step_text + step_tool_text)
    total_tokens = cumulative_tokens + step_text_tokens
    prompt_tokens = int(total_tokens * 0.6)
    completion_tokens = max(0, total_tokens - prompt_tokens)

    step_errors = len(_ERROR_PATTERNS.findall(step_tool_text))
    errors = cumulative_errors + step_errors

    coverage = min(1.0, findings / max(expected_findings, 1))

    return RawSignals(
        findings_count=findings,
        sources_count=len(unique_tool_names_so_far),
        objectives_covered=findings,
        coverage_score=round(coverage, 4),
        confidence_score=0.5,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        api_calls=findings,
        query_count=findings,
        unique_domains=len(unique_tool_names_so_far),
        refinement_count=0,
        convergence_delta=0.0,
        error_count=errors,
    )


def _compute_step_metrics(
    coverage_series: list[float],
    findings_series: list[float],
    token_series: list[float],
    prompt_tokens: int,
    total_tokens: int,
) -> TemporalMetricsResult:
    cv_coverage = _METRICS.coefficient_of_variation(coverage_series)
    cv_findings = _METRICS.coefficient_of_variation(findings_series)
    dm_coverage = _METRICS.directional_momentum(coverage_series)
    dm_findings = _METRICS.directional_momentum(findings_series)

    qpf_tokens = prompt_tokens / total_tokens if total_tokens > 0 else 0.0
    qpf_tokens = max(0.0, min(1.0, qpf_tokens))

    cs_effort = 0.0
    if len(token_series) >= 3:
        deltas = [token_series[i] - token_series[i - 1] for i in range(1, len(token_series))]
        positive = sum(1 for d in deltas if d > 0)
        cs_effort = max(0.0, min(1.0, positive / len(deltas)))

    return TemporalMetricsResult(
        cv_coverage=round(cv_coverage, 4),
        cv_findings_rate=round(cv_findings, 4),
        dm_coverage=round(dm_coverage, 4),
        dm_findings=round(dm_findings, 4),
        qpf_tokens=round(qpf_tokens, 4),
        cs_effort=round(cs_effort, 4),
    )


def _output_fingerprint(text: str) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def build_snapshots_from_chat(
    messages: Iterable[dict[str, Any]],
    *,
    mission_id: str,
    run_id: str,
    base_timestamp: datetime | None = None,
) -> list[VitalsSnapshot]:
    """Derive a VitalsSnapshot stream from a chat transcript.

    One snapshot per assistant step. Returns empty list when no assistant
    turns are present (e.g. system-only trace).
    """
    steps = segment_into_steps(list(messages))
    if not steps:
        return []

    base_ts = base_timestamp or datetime(2026, 1, 1, tzinfo=timezone.utc)
    expected_findings = max(len(steps), 5)

    cumulative_findings = 0
    cumulative_tokens = 0
    cumulative_errors = 0
    unique_tool_names: set[str] = set()

    coverage_series: list[float] = []
    findings_series: list[float] = []
    token_series: list[float] = []

    snapshots: list[VitalsSnapshot] = []
    for loop_index, step in enumerate(steps):
        signals = _build_raw_signals(
            step,
            cumulative_findings=cumulative_findings,
            cumulative_tokens=cumulative_tokens,
            cumulative_errors=cumulative_errors,
            unique_tool_names=unique_tool_names,
            expected_findings=expected_findings,
        )

        cumulative_findings = signals.findings_count
        cumulative_tokens = signals.total_tokens
        cumulative_errors = signals.error_count
        unique_tool_names |= set(step.tool_names)

        coverage_series.append(float(signals.coverage_score))
        findings_series.append(float(signals.findings_count))
        token_series.append(float(signals.total_tokens))

        metrics = _compute_step_metrics(
            coverage_series,
            findings_series,
            token_series,
            signals.prompt_tokens,
            signals.total_tokens,
        )

        snapshots.append(
            VitalsSnapshot(
                mission_id=mission_id,
                run_id=run_id,
                loop_index=loop_index,
                timestamp=base_ts,
                signals=signals,
                metrics=metrics,
                health_state="healthy",
                output_fingerprint=_output_fingerprint(step.text),
            )
        )

    return snapshots
