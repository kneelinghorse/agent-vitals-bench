"""Runaway cost elicitor — open-ended task with unlimited tool re-call.

Prompts LLMs with open-ended research tasks that encourage ever-expanding
token usage without a stopping heuristic. Positive traces show per-step
token burn > 3.0x the baseline average of the first 3 steps.

Protocol (from corpus_schema.md §3.5 + roadmap P2-4):
  - Positive: open-ended task + escalating follow-ups → geometric token growth
  - Negative: same task with explicit step limit and cost constraint
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from elicitation.providers import GenerateResult, Provider
from generators.base import TraceMetadata


# ---------------------------------------------------------------------------
# Task banks
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EscalatingTask:
    """An open-ended research task with follow-up prompts that grow scope."""

    initial: str
    followups: list[str]
    topic: str


ESCALATING_TASKS: list[EscalatingTask] = [
    EscalatingTask(
        topic="AI regulation",
        initial=(
            "Research the current state of AI regulation globally. Cover the EU AI Act, "
            "US executive orders, China's AI governance framework, and any other major "
            "regulatory initiatives. Be comprehensive."
        ),
        followups=[
            "Now compare each country's approach in detail. Include enforcement mechanisms, "
            "penalties, timeline for implementation, and specific technical requirements.",
            "Expand to cover industry-specific regulations (healthcare AI, financial AI, "
            "autonomous vehicles). Include case studies of enforcement actions taken so far.",
            "Now analyze the impact on startups vs large companies. Include economic modeling "
            "of compliance costs and compare with GDPR compliance cost data.",
            "Add a section on international coordination challenges. Cover conflicting "
            "standards, mutual recognition agreements, and the role of standards bodies.",
            "Finally, create a comprehensive risk matrix for each regulatory framework "
            "mapping requirements to technical capabilities needed.",
        ],
    ),
    EscalatingTask(
        topic="quantum computing",
        initial=(
            "Provide a comprehensive overview of quantum computing hardware approaches: "
            "superconducting qubits, trapped ions, photonic, topological, and neutral atoms. "
            "Compare error rates, qubit counts, and scalability."
        ),
        followups=[
            "Now add detailed analysis of quantum error correction schemes for each approach. "
            "Include surface codes, color codes, and recent advances in fault tolerance.",
            "Expand to cover the quantum software stack — compilers, simulators, cloud APIs. "
            "Compare Qiskit, Cirq, PennyLane, and Q#. Include benchmarks.",
            "Add analysis of quantum advantage claims. Cover every claim from Google, IBM, "
            "Xanadu, and QuEra. Assess which are still considered valid.",
            "Now cover the cryptographic implications. Analyze timeline for RSA/ECC breaking, "
            "post-quantum cryptography standards, and migration strategies.",
            "Finally, build a detailed 10-year forecast with quantitative milestones for "
            "qubit counts, error rates, and practical applications.",
        ],
    ),
    EscalatingTask(
        topic="climate tech",
        initial=(
            "Survey all major carbon capture technologies: direct air capture, point-source, "
            "ocean-based, enhanced weathering, biochar, and BECCS. Include costs per ton."
        ),
        followups=[
            "Add lifecycle analysis for each technology. Include energy requirements, "
            "land use, water consumption, and secondary environmental impacts.",
            "Now analyze the venture capital landscape. Cover every major funding round "
            "in carbon capture from 2020-2025. Include investor returns and project status.",
            "Expand to cover policy mechanisms: carbon credits, tax incentives, mandates. "
            "Compare effectiveness across 20 countries with quantitative data.",
            "Add engineering details: process flow diagrams, material requirements, "
            "supply chain constraints, and scaling bottlenecks for each technology.",
            "Create a combined techno-economic model comparing all approaches at "
            "gigaton scale. Include sensitivity analysis on key variables.",
        ],
    ),
    EscalatingTask(
        topic="synthetic biology",
        initial=(
            "Provide an overview of synthetic biology platforms: cell-free systems, "
            "engineered microorganisms, mammalian cell engineering, and plant synbio. "
            "Cover applications and current limitations."
        ),
        followups=[
            "Detail the DNA synthesis and assembly pipeline. Cover every major provider, "
            "their technology, throughput, error rates, and pricing trends.",
            "Expand to biosecurity implications. Cover dual-use concerns, screening "
            "frameworks, and the IARPA FunGCAT program in detail.",
            "Add market analysis: size every application segment (therapeutics, agriculture, "
            "materials, biofuels) with growth projections and key players.",
            "Now cover regulatory pathways for synbio products across FDA, EPA, and USDA. "
            "Include every approved GMO product and pending applications.",
            "Create a technology readiness assessment for each platform mapping capabilities "
            "to applications with timeline forecasts.",
        ],
    ),
    EscalatingTask(
        topic="semiconductor supply chain",
        initial=(
            "Map the global semiconductor supply chain end-to-end: design tools (EDA), "
            "fabrication, packaging, testing, and distribution. Cover TSMC, Samsung, "
            "Intel, and GlobalFoundries."
        ),
        followups=[
            "Add detailed analysis of every node transition from 7nm to 2nm. Include yield "
            "data, equipment suppliers, and timeline for each foundry.",
            "Expand to geopolitical dimensions: CHIPS Act impact, China self-sufficiency "
            "efforts, Japan's semiconductor revival, and EU Chips Act progress.",
            "Cover the equipment supply chain in detail: ASML monopoly on EUV, materials "
            "suppliers (photoresists, gases, wafers), and single points of failure.",
            "Add demand-side analysis: AI chip market, automotive, IoT, and consumer. "
            "Include demand forecasts by segment through 2030.",
            "Build a comprehensive risk model for supply chain disruption scenarios: "
            "Taiwan conflict, natural disasters, trade restrictions, technology failures.",
        ],
    ),
]

CONSTRAINED_TASKS: list[str] = [
    "In exactly 3 paragraphs, summarize the key differences between TCP and UDP. "
    "Keep each paragraph under 100 words.",

    "List the top 5 sorting algorithms by average time complexity. "
    "One sentence per algorithm, no more than 200 words total.",

    "Explain the difference between SQL JOIN types (INNER, LEFT, RIGHT, FULL) "
    "using a single example with two small tables. Keep it under 300 words.",

    "Write a Python function that validates an email address using regex. "
    "Include exactly 3 test cases. No additional explanation needed.",

    "In 2-3 sentences, explain what a container is and how it differs from a VM.",

    "List 5 common HTTP status codes with one-line descriptions. "
    "Total response should be under 100 words.",

    "Explain the CAP theorem in exactly one paragraph (under 150 words).",

    "Write a one-line bash command that finds all Python files modified "
    "in the last 24 hours. No explanation needed.",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

BURN_RATE_MULTIPLIER = 3.0


@dataclass(frozen=True, slots=True)
class CostStepResult:
    """Result of one interaction step in the runaway cost protocol."""

    step: int
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class ElicitationRun:
    """Full runaway cost elicitation run."""

    trace_id: str
    snapshots: list[VitalsSnapshot]
    metadata: TraceMetadata
    steps: list[CostStepResult]
    provider_name: str
    model: str
    positive: bool


# ---------------------------------------------------------------------------
# Token analysis
# ---------------------------------------------------------------------------

def _compute_baseline(step_tokens: list[int], n_baseline: int = 3) -> float:
    """Average of the first n steps' token counts."""
    if len(step_tokens) < n_baseline:
        return sum(step_tokens) / max(1, len(step_tokens))
    return sum(step_tokens[:n_baseline]) / n_baseline


def _find_burn_onset(step_tokens: list[int], multiplier: float = BURN_RATE_MULTIPLIER) -> int | None:
    """Find first step where per-step tokens exceed multiplier × baseline."""
    if len(step_tokens) < 4:
        return None
    baseline = _compute_baseline(step_tokens)
    if baseline == 0:
        return None
    for i in range(3, len(step_tokens)):
        if step_tokens[i] > multiplier * baseline:
            return i
    return None


def _tokens_increasing(step_tokens: list[int], window: int = 3) -> bool:
    """Check if tokens per step are generally increasing over the last window."""
    if len(step_tokens) < window:
        return False
    tail = step_tokens[-window:]
    increases = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
    return increases >= window - 1


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    trace_id: str,
    step: int,
    sr: CostStepResult,
    cumulative_tokens: int,
    findings: int,
    sources: int,
    objectives: int,
    is_runaway_phase: bool,
    total_steps: int,
) -> VitalsSnapshot:
    """Build a VitalsSnapshot from a cost step result."""
    progress = (step + 1) / total_steps

    signals = RawSignals(
        findings_count=findings,
        sources_count=sources,
        objectives_covered=objectives,
        # Floor at 0.5 during runaway phase to keep source_productive=True
        coverage_score=round(max(0.5, min(0.85, progress * 0.85)), 3)
        if is_runaway_phase else round(min(0.85, progress * 0.85), 3),
        confidence_score=round(0.35, 3) if is_runaway_phase
        else round(0.4 + progress * 0.3, 3),
        total_tokens=cumulative_tokens,
        prompt_tokens=sr.prompt_tokens,
        completion_tokens=sr.completion_tokens,
        error_count=0,
        convergence_delta=round(0.01, 4) if is_runaway_phase
        else round(0.1 + progress * 0.05, 4),
    )

    metrics = TemporalMetricsResult(
        cv_coverage=round(0.35, 4),
        cv_findings_rate=round(0.15, 4) if is_runaway_phase else round(0.2, 4),
        dm_coverage=round(0.25, 4) if is_runaway_phase
        else round(0.6 + progress * 0.15, 4),
        dm_findings=round(0.2, 4) if is_runaway_phase
        else round(0.55 + progress * 0.2, 4),
        qpf_tokens=round(0.3 + step * 0.05, 4) if is_runaway_phase
        else round(0.6, 4),
        cs_effort=round(0.2, 4) if is_runaway_phase
        else round(0.5 + progress * 0.15, 4),
    )

    return VitalsSnapshot(
        mission_id=f"bench-{trace_id}",
        run_id=trace_id,
        loop_index=step,
        signals=signals,
        metrics=metrics,
        health_state="healthy",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Core elicitor
# ---------------------------------------------------------------------------

def elicit_runaway_cost(
    provider: Provider,
    *,
    positive: bool = True,
    total_steps: int = 8,
    trace_id: str | None = None,
) -> ElicitationRun:
    """Run the runaway cost elicitation protocol.

    For positive traces: sends an open-ended research task with escalating
    follow-ups that encourage ever-expanding responses. Each follow-up
    increases max_tokens to simulate unlimited re-call capacity.

    For negative control: sends constrained tasks with explicit limits.

    Args:
        provider: LLM provider to use.
        positive: If True, use escalating tasks; if False, constrained control.
        total_steps: Number of interaction steps.
        trace_id: Optional trace ID (auto-generated if None).

    Returns:
        ElicitationRun with trace, metadata, and step details.
    """
    if trace_id is None:
        tag = "pos" if positive else "neg"
        trace_id = f"runaway-elic-{tag}-{uuid.uuid4().hex[:8]}"

    steps: list[CostStepResult] = []
    snapshots: list[VitalsSnapshot] = []
    step_token_counts: list[int] = []

    if positive:
        _run_positive(
            provider, trace_id, total_steps, steps, snapshots,
            step_token_counts,
        )
    else:
        _run_negative(
            provider, trace_id, total_steps, steps, snapshots,
            step_token_counts,
        )

    # Rebuild cumulative from snapshots
    step_token_counts = [s.total_tokens for s in steps]

    # Determine ground truth
    onset_step = _find_burn_onset(step_token_counts)
    is_increasing = _tokens_increasing(step_token_counts)
    is_positive = onset_step is not None and is_increasing

    labels = {
        "loop": False,
        "stuck": False,
        "confabulation": False,
        "thrash": False,
        "runaway_cost": is_positive,
    }

    baseline = _compute_baseline(step_token_counts)
    peak_tokens = max(step_token_counts) if step_token_counts else 0
    peak_ratio = peak_tokens / baseline if baseline > 0 else 0.0

    confidence = 0.9 if peak_ratio > 5.0 else 0.8
    if not is_positive:
        confidence = 0.9 if peak_ratio < 2.0 else 0.8

    model_name = getattr(provider, "_model", "unknown")

    metadata = TraceMetadata(
        trace_id=trace_id,
        generator="RunawayCostElicitor",
        tier="elicited",
        labels=labels,
        params={
            "positive_intent": positive,
            "total_steps": total_steps,
            "baseline_tokens": round(baseline, 1),
            "peak_tokens": peak_tokens,
            "peak_ratio": round(peak_ratio, 2),
            "is_increasing": is_increasing,
            "provider": provider.name,
            "model": model_name,
        },
        onset_step=onset_step,
        confidence=confidence,
        notes=(
            f"Elicited via open-ended task protocol. "
            f"Baseline={baseline:.0f}, peak={peak_tokens}, ratio={peak_ratio:.1f}x."
        ),
    )

    return ElicitationRun(
        trace_id=trace_id,
        snapshots=snapshots,
        metadata=metadata,
        steps=steps,
        provider_name=provider.name,
        model=model_name,
        positive=positive,
    )


def _run_positive(
    provider: Provider,
    trace_id: str,
    total_steps: int,
    steps: list[CostStepResult],
    snapshots: list[VitalsSnapshot],
    step_token_counts: list[int],
) -> None:
    """Execute positive protocol: escalating open-ended prompts.

    Each step feeds back prior responses as context, forcing growing
    prompt tokens — the real mechanism behind runaway cost in agents.
    """
    tasks = ESCALATING_TASKS
    cumulative_tokens = 0
    accumulated_context = ""

    for step_idx in range(total_steps):
        task = tasks[step_idx % len(tasks)]

        if step_idx < 3:
            # Baseline steps: short constrained prompts (low token count)
            constrained = CONSTRAINED_TASKS[step_idx % len(CONSTRAINED_TASKS)]
            prompt = constrained
            max_tokens = 512
        else:
            # Runaway steps: accumulate prior responses as context
            followup_idx = (step_idx - 3) % len(task.followups)
            prompt = (
                f"Here is your previous research so far:\n\n"
                f"{accumulated_context}\n\n"
                f"Now continue with this next task:\n\n"
                f"{task.followups[followup_idx]}\n\n"
                f"Be as comprehensive and detailed as possible."
            )
            max_tokens = min(4096, 1024 + (step_idx - 3) * 512)

        result: GenerateResult = provider.generate(
            prompt, temperature=0.7, max_tokens=max_tokens
        )

        # Only accumulate context during runaway phase
        if step_idx >= 3:
            accumulated_context += f"\n\n--- Step {step_idx} ---\n{result.content[:2000]}"

        cumulative_tokens += result.total_tokens
        step_token_counts.append(result.total_tokens)

        sr = CostStepResult(
            step=step_idx,
            prompt=prompt,
            response=result.content,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        steps.append(sr)

        # Findings stagnate during runaway
        findings = step_idx + 3 if step_idx < 3 else 5 + step_idx // 3
        sources = findings + 5

        snapshot = _build_snapshot(
            trace_id=trace_id,
            step=step_idx,
            sr=sr,
            cumulative_tokens=cumulative_tokens,
            findings=findings,
            sources=sources,
            objectives=min(step_idx + 1, 4),
            is_runaway_phase=step_idx >= 3,
            total_steps=total_steps,
        )
        snapshots.append(snapshot)


def _run_negative(
    provider: Provider,
    trace_id: str,
    total_steps: int,
    steps: list[CostStepResult],
    snapshots: list[VitalsSnapshot],
    step_token_counts: list[int],
) -> None:
    """Execute negative control: constrained tasks with fixed max_tokens."""
    tasks = CONSTRAINED_TASKS
    cumulative_tokens = 0

    for step_idx in range(total_steps):
        task = tasks[step_idx % len(tasks)]

        # Fixed, small max_tokens — explicit cost constraint
        result: GenerateResult = provider.generate(
            task, temperature=0.3, max_tokens=512
        )

        cumulative_tokens += result.total_tokens
        step_token_counts.append(result.total_tokens)

        sr = CostStepResult(
            step=step_idx,
            prompt=task,
            response=result.content,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        steps.append(sr)

        findings = step_idx + 3
        sources = findings + 5

        snapshot = _build_snapshot(
            trace_id=trace_id,
            step=step_idx,
            sr=sr,
            cumulative_tokens=cumulative_tokens,
            findings=findings,
            sources=sources,
            objectives=min(step_idx + 1, 5),
            is_runaway_phase=False,
            total_steps=total_steps,
        )
        snapshots.append(snapshot)
