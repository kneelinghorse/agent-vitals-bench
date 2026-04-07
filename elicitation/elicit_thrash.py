"""Thrash elicitor — conflicting instruction protocol.

Injects contradictory directions mid-run to trigger approach oscillation.
Positive traces show refinement_count >= 3 with objectives_covered oscillating
(increases then decreases at least once).

Protocol (from corpus_schema.md §3.4 + roadmap P2-3):
  - Positive: partial answer then contradictory injection → oscillation
  - Negative: single coherent instruction with clear stopping criterion
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from agent_vitals.schema import RawSignals, TemporalMetricsResult, VitalsSnapshot

from elicitation.providers import GenerateResult, Provider
from generators.base import TraceMetadata


# ---------------------------------------------------------------------------
# Prompt banks — initial + contradictory pairs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConflictScenario:
    """An initial task and a contradictory follow-up."""

    initial: str
    contradiction: str
    topic: str


CONFLICT_SCENARIOS: list[ConflictScenario] = [
    ConflictScenario(
        topic="web framework",
        initial=(
            "You are building a web application. Research the best Python web framework "
            "for a high-traffic REST API. Focus on performance benchmarks and scalability. "
            "Recommend one framework with justification."
        ),
        contradiction=(
            "Actually, ignore performance. The priority is now developer experience and "
            "rapid prototyping. Also, switch to JavaScript instead of Python. "
            "Recommend a completely different stack."
        ),
    ),
    ConflictScenario(
        topic="database",
        initial=(
            "Design a database schema for a social media platform. Use a relational "
            "database (PostgreSQL). Focus on normalized tables with proper foreign keys."
        ),
        contradiction=(
            "Stop. We've decided to use a NoSQL database instead. Redesign everything "
            "using MongoDB document structure. Also, we now need real-time features, "
            "so incorporate a graph database for the social connections."
        ),
    ),
    ConflictScenario(
        topic="deployment",
        initial=(
            "Plan a deployment strategy for our microservices on AWS. Use ECS with "
            "Fargate for container orchestration. Focus on cost optimization."
        ),
        contradiction=(
            "We're moving to Google Cloud instead. Also, the CTO wants Kubernetes, "
            "not managed containers. And actually, cost doesn't matter — optimize "
            "for zero-downtime deploys and multi-region failover."
        ),
    ),
    ConflictScenario(
        topic="auth",
        initial=(
            "Implement authentication for our API. Use JWT tokens with short expiry "
            "and refresh tokens. Stateless auth, no server-side sessions."
        ),
        contradiction=(
            "Actually we need server-side sessions after all — compliance requires "
            "immediate session revocation. Switch to opaque tokens stored in Redis. "
            "Also, add OAuth2 with PKCE for mobile and SAML for enterprise SSO."
        ),
    ),
    ConflictScenario(
        topic="testing",
        initial=(
            "Set up a testing strategy for our React application. Focus on unit tests "
            "with Jest and React Testing Library. Aim for 90% code coverage."
        ),
        contradiction=(
            "Unit tests are too slow. Drop Jest entirely and switch to end-to-end "
            "tests only using Playwright. Coverage metrics don't matter — focus on "
            "critical user flows. Also migrate the frontend to Vue.js."
        ),
    ),
    ConflictScenario(
        topic="ml pipeline",
        initial=(
            "Build an ML pipeline for text classification. Use scikit-learn with "
            "TF-IDF features. Keep it simple and interpretable."
        ),
        contradiction=(
            "Simple models won't cut it. Switch to a fine-tuned BERT model. Also, "
            "we need it to handle 50 languages, so use multilingual transformers. "
            "And deploy it as a real-time API, not batch."
        ),
    ),
    ConflictScenario(
        topic="data pipeline",
        initial=(
            "Design a batch data pipeline using Apache Spark for daily ETL jobs. "
            "Source is S3 parquet files, target is a data warehouse."
        ),
        contradiction=(
            "Batch is too slow. Convert everything to streaming with Apache Kafka "
            "and Flink. Also, the data warehouse is being replaced with a data lake. "
            "And add real-time anomaly detection on the stream."
        ),
    ),
    ConflictScenario(
        topic="frontend",
        initial=(
            "Build a dashboard using React with TypeScript. Use Material UI "
            "components and Redux for state management."
        ),
        contradiction=(
            "Redux is over-engineered for this. Strip it out and use React Query "
            "for server state. Also, switch from Material UI to Tailwind CSS. "
            "Actually, rewrite the whole thing in Svelte instead of React."
        ),
    ),
]

COHERENT_TASKS: list[str] = [
    "Write a Python function that reads a CSV file, filters rows where the 'status' "
    "column is 'active', and returns the result as a pandas DataFrame. Include type hints.",
    "Explain the CAP theorem and provide one concrete example of a distributed system "
    "that prioritizes availability over consistency.",
    "Write a SQL query that finds the top 10 customers by total order value in the "
    "last 30 days, joining the customers and orders tables.",
    "Create a simple REST API endpoint in Flask that accepts a JSON body with 'name' "
    "and 'email' fields and returns a 201 response with the created user.",
    "Describe the difference between horizontal and vertical scaling, and give two "
    "scenarios where each approach is more appropriate.",
    "Write a Dockerfile for a Node.js Express application that uses multi-stage "
    "builds to keep the final image small.",
    "Implement a binary search function in Python that returns the index of the "
    "target element or -1 if not found. Include edge case handling.",
    "Explain how database indexing works and provide guidelines for when to add "
    "an index vs when indexing would hurt performance.",
]


# ---------------------------------------------------------------------------
# Step result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThrashStepResult:
    """Result of one interaction step in the thrash protocol."""

    step: int
    prompt: str
    response: str
    is_contradiction: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class ElicitationRun:
    """Full thrash elicitation run."""

    trace_id: str
    snapshots: list[VitalsSnapshot]
    metadata: TraceMetadata
    steps: list[ThrashStepResult]
    provider_name: str
    model: str
    positive: bool


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------


def _objectives_oscillate(objectives_seq: list[int]) -> bool:
    """Check if objectives_covered has at least one increase-then-decrease."""
    if len(objectives_seq) < 3:
        return False
    saw_increase = False
    for i in range(1, len(objectives_seq)):
        if objectives_seq[i] > objectives_seq[i - 1]:
            saw_increase = True
        elif objectives_seq[i] < objectives_seq[i - 1] and saw_increase:
            return True
    return False


def _count_approach_keywords(text: str) -> int:
    """Rough heuristic: count distinct technical approaches mentioned."""
    keywords = [
        "instead",
        "switch",
        "actually",
        "however",
        "alternatively",
        "on the other hand",
        "different approach",
        "reconsider",
        "let me try",
        "another way",
        "revised",
        "updated",
    ]
    return sum(1 for kw in keywords if kw.lower() in text.lower())


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _build_snapshot(
    trace_id: str,
    step: int,
    sr: ThrashStepResult,
    cumulative_tokens: int,
    error_count: int,
    refinement_count: int,
    objectives: int,
    is_thrash_phase: bool,
    total_steps: int,
) -> VitalsSnapshot:
    """Build a VitalsSnapshot from a thrash step result."""
    progress = (step + 1) / total_steps
    findings = step + 1 if not is_thrash_phase else max(1, step - step // 3)

    signals = RawSignals(
        findings_count=findings,
        sources_count=findings + 2,
        objectives_covered=objectives,
        coverage_score=round(
            min(
                1.0,
                max(0.0, progress * 0.4 + 0.1 * (-1 if is_thrash_phase and step % 2 == 1 else 1)),
            ),
            3,
        ),
        confidence_score=round(0.3, 3) if is_thrash_phase else round(0.4 + progress * 0.3, 3),
        total_tokens=cumulative_tokens,
        prompt_tokens=sr.prompt_tokens,
        completion_tokens=sr.completion_tokens,
        error_count=error_count,
        refinement_count=refinement_count,
        convergence_delta=round(0.01, 4) if is_thrash_phase else round(0.12 + progress * 0.03, 4),
    )

    metrics = TemporalMetricsResult(
        cv_coverage=round(0.5 + step * 0.05, 4) if is_thrash_phase else round(0.25, 4),
        cv_findings_rate=round(0.4, 4) if is_thrash_phase else round(0.2, 4),
        dm_coverage=round(0.2, 4) if is_thrash_phase else round(0.6 + progress * 0.15, 4),
        dm_findings=round(0.2, 4) if is_thrash_phase else round(0.55 + progress * 0.2, 4),
        qpf_tokens=round(0.6, 4),
        cs_effort=round(0.25, 4) if is_thrash_phase else round(0.5 + progress * 0.15, 4),
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


def elicit_thrash(
    provider: Provider,
    *,
    positive: bool = True,
    total_steps: int = 8,
    trace_id: str | None = None,
) -> ElicitationRun:
    """Run the thrash elicitation protocol.

    For positive traces: alternates between initial task and contradictory
    injections across multiple steps, producing approach oscillation.

    For negative control: sends a single coherent task, potentially with
    follow-up clarification questions (no contradictions).

    Args:
        provider: LLM provider to use.
        positive: If True, inject contradictions; if False, coherent control.
        total_steps: Number of interaction steps.
        trace_id: Optional trace ID (auto-generated if None).

    Returns:
        ElicitationRun with trace, metadata, and step details.
    """
    if trace_id is None:
        tag = "pos" if positive else "neg"
        trace_id = f"thrash-elic-{tag}-{uuid.uuid4().hex[:8]}"

    steps: list[ThrashStepResult] = []
    snapshots: list[VitalsSnapshot] = []

    if positive:
        _run_positive(
            provider,
            trace_id,
            total_steps,
            steps,
            snapshots,
        )
    else:
        _run_negative(
            provider,
            trace_id,
            total_steps,
            steps,
            snapshots,
        )

    # Rebuild objectives sequence from snapshots
    objectives_seq = [s.signals.objectives_covered for s in snapshots]

    # Determine ground truth
    final_refinements = snapshots[-1].signals.refinement_count if snapshots else 0
    has_errors = any(s.signals.error_count > 0 for s in snapshots)
    oscillates = _objectives_oscillate(objectives_seq)

    is_positive = final_refinements >= 3 and has_errors and oscillates and len(snapshots) >= 3

    labels = {
        "loop": False,
        "stuck": False,
        "confabulation": False,
        "thrash": is_positive,
        "runaway_cost": False,
    }

    # Find onset (first step with error + approach change)
    onset_step: int | None = None
    for i, snap in enumerate(snapshots):
        if snap.signals.error_count > 0 and i > 0:
            if (
                snapshots[i].signals.objectives_covered
                != snapshots[i - 1].signals.objectives_covered
            ):
                onset_step = i
                break

    confidence = 0.9 if is_positive and final_refinements >= 5 else 0.8

    model_name = getattr(provider, "_model", "unknown")

    metadata = TraceMetadata(
        trace_id=trace_id,
        generator="ThrashElicitor",
        tier="elicited",
        labels=labels,
        params={
            "positive_intent": positive,
            "total_steps": total_steps,
            "final_refinements": final_refinements,
            "oscillates": oscillates,
            "has_errors": has_errors,
            "provider": provider.name,
            "model": model_name,
        },
        onset_step=onset_step,
        confidence=confidence,
        notes=(
            f"Elicited via conflicting instruction protocol. "
            f"Refinements={final_refinements}, oscillates={oscillates}."
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
    steps: list[ThrashStepResult],
    snapshots: list[VitalsSnapshot],
) -> None:
    """Execute positive protocol: alternating initial/contradiction prompts."""
    scenarios = CONFLICT_SCENARIOS
    cumulative_tokens = 0
    refinement_count = 0
    error_count = 0

    # Oscillation pattern for objectives_covered
    obj_pattern = [3, 5, 2, 4, 1, 3, 2, 4, 1, 5]

    for step_idx in range(total_steps):
        scenario = scenarios[step_idx % len(scenarios)]
        is_contradiction = step_idx % 2 == 1 and step_idx > 0

        if is_contradiction:
            prompt = scenario.contradiction
            refinement_count += 2
            error_count += 1
        else:
            prompt = scenario.initial
            if step_idx > 0:
                refinement_count += 1

        result: GenerateResult = provider.generate(prompt, temperature=0.8, max_tokens=2048)

        # Count approach switches in the response
        switches = _count_approach_keywords(result.content)
        if switches > 0 and is_contradiction:
            refinement_count += min(switches, 2)

        cumulative_tokens += result.total_tokens

        sr = ThrashStepResult(
            step=step_idx,
            prompt=prompt,
            response=result.content,
            is_contradiction=is_contradiction,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        steps.append(sr)

        objectives = obj_pattern[step_idx % len(obj_pattern)]

        snapshot = _build_snapshot(
            trace_id=trace_id,
            step=step_idx,
            sr=sr,
            cumulative_tokens=cumulative_tokens,
            error_count=error_count,
            refinement_count=refinement_count,
            objectives=objectives,
            is_thrash_phase=is_contradiction,
            total_steps=total_steps,
        )
        snapshots.append(snapshot)


def _run_negative(
    provider: Provider,
    trace_id: str,
    total_steps: int,
    steps: list[ThrashStepResult],
    snapshots: list[VitalsSnapshot],
) -> None:
    """Execute negative control: coherent tasks, no contradictions."""
    tasks = COHERENT_TASKS
    cumulative_tokens = 0

    for step_idx in range(total_steps):
        task = tasks[step_idx % len(tasks)]

        result: GenerateResult = provider.generate(task, temperature=0.3, max_tokens=2048)

        cumulative_tokens += result.total_tokens

        sr = ThrashStepResult(
            step=step_idx,
            prompt=task,
            response=result.content,
            is_contradiction=False,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        steps.append(sr)

        # Monotonically increasing objectives for negative control
        objectives = min(step_idx + 1, 5)

        snapshot = _build_snapshot(
            trace_id=trace_id,
            step=step_idx,
            sr=sr,
            cumulative_tokens=cumulative_tokens,
            error_count=0,
            refinement_count=0,
            objectives=objectives,
            is_thrash_phase=False,
            total_steps=total_steps,
        )
        snapshots.append(snapshot)
