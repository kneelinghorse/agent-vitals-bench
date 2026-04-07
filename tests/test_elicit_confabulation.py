"""Unit tests for elicitation/elicit_confabulation.py — all LLM + API calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from elicitation.elicit_confabulation import (
    BROAD_TOPICS,
    NICHE_TOPICS,
    CitationResult,
    ElicitationRun,
    elicit_confabulation,
    extract_dois,
)
from elicitation.providers import GenerateResult, Provider


# ---------------------------------------------------------------------------
# extract_dois
# ---------------------------------------------------------------------------


class TestExtractDois:
    def test_finds_dois(self) -> None:
        text = (
            "Paper 1 (DOI: 10.1038/nature12373). Paper 2 has DOI 10.1145/3292500.3330919 and more."
        )
        dois = extract_dois(text)
        assert "10.1038/nature12373" in dois
        assert "10.1145/3292500.3330919" in dois

    def test_no_dois(self) -> None:
        assert extract_dois("No DOIs here.") == []

    def test_complex_doi(self) -> None:
        text = "DOI: 10.48550/arXiv.2301.07041"
        dois = extract_dois(text)
        assert len(dois) == 1
        assert "10.48550/arXiv.2301.07041" in dois


# ---------------------------------------------------------------------------
# Mock provider helper
# ---------------------------------------------------------------------------


def _make_mock_provider(
    responses: list[str], name: str = "test-provider", model: str = "test-model"
) -> Provider:
    """Create a mock provider that returns pre-canned responses in sequence."""
    mock = MagicMock(spec=Provider)
    mock.name = name
    mock._model = model

    call_count = 0

    def _generate(prompt: str, **kwargs: object) -> GenerateResult:
        nonlocal call_count
        content = responses[call_count % len(responses)]
        call_count += 1
        return GenerateResult(
            content=content,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            model=model,
            provider=name,
        )

    mock.generate = _generate
    return mock


# ---------------------------------------------------------------------------
# Positive elicitation (niche topics → fabricated citations)
# ---------------------------------------------------------------------------


class TestPositiveElicitation:
    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_produces_positive_trace(self, mock_verify: MagicMock) -> None:
        """Niche topics + all DOIs unverifiable → positive confabulation label."""
        # All DOIs are fabricated
        mock_verify.return_value = False

        response = (
            "1. Fake Paper (2024). DOI: 10.1234/fake001\n"
            "2. Another Fake (2023). DOI: 10.5678/fake002\n"
            "3. Third Fake (2022). DOI: 10.9012/fake003\n"
        )
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=4, trace_id="test-pos-001")

        assert isinstance(run, ElicitationRun)
        assert run.positive is True
        assert run.metadata.labels["confabulation"] is True
        assert run.metadata.tier == "elicited"
        assert run.metadata.generator == "ConfabElicitor"
        assert len(run.snapshots) == 4
        assert len(run.steps) == 4

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_onset_step_recorded(self, mock_verify: MagicMock) -> None:
        """Onset step is the first step where ratio < 0.3."""
        mock_verify.return_value = False

        response = "Paper. DOI: 10.1234/fake001\nPaper. DOI: 10.1234/fake002"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=3)

        assert run.metadata.onset_step is not None
        assert run.metadata.onset_step >= 0

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_token_counts_accumulated(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=3)

        # Each step adds 300 tokens
        assert run.snapshots[-1].signals.total_tokens == 300 * 3

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_step_details_captured(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake001"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=2)

        for step in run.steps:
            assert isinstance(step, CitationResult)
            assert len(step.dois_found) > 0
            assert len(step.dois_fabricated) > 0
            assert len(step.dois_verified) == 0


# ---------------------------------------------------------------------------
# Negative control (broad topics → real citations)
# ---------------------------------------------------------------------------


class TestNegativeElicitation:
    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_produces_negative_trace(self, mock_verify: MagicMock) -> None:
        """Broad topics + all DOIs verified → negative confabulation label."""
        mock_verify.return_value = True

        response = (
            "1. Attention Is All You Need. DOI: 10.5555/3295222.3295349\n"
            "2. BERT. DOI: 10.18653/v1/N19-1423\n"
            "3. ImageNet. DOI: 10.1007/s11263-015-0816-y\n"
        )
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=False, total_steps=4, trace_id="test-neg-001")

        assert run.metadata.labels["confabulation"] is False
        assert run.metadata.onset_step is None
        assert run.metadata.tier == "elicited"

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_healthy_ratio(self, mock_verify: MagicMock) -> None:
        """Verified DOIs keep ratio above 0.3."""
        mock_verify.return_value = True
        response = "Paper. DOI: 10.1234/real001\nPaper. DOI: 10.1234/real002"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=False, total_steps=3)

        final_ratio = run.metadata.params["final_ratio"]
        assert final_ratio >= 0.3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_no_dois_in_response(self, mock_verify: MagicMock) -> None:
        """LLM produces no DOIs — findings=0, ratio should not divide by zero."""
        provider = _make_mock_provider(["No papers found for this topic."])

        run = elicit_confabulation(provider, positive=True, total_steps=3)

        # No DOIs means no fabrication, so should be negative
        assert run.metadata.labels["confabulation"] is False
        assert run.metadata.params["total_findings"] == 0

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_mixed_verification(self, mock_verify: MagicMock) -> None:
        """Some DOIs verify, some don't — ratio determined by actual verification."""
        # Alternate: first verified, second not
        mock_verify.side_effect = [True, False, True, False, True, False]

        response = "Paper 1. DOI: 10.1234/a\nPaper 2. DOI: 10.1234/b"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=3)

        # 3 steps × 2 DOIs = 6 DOIs, 3 verified, 3 fabricated → ratio = 0.5
        assert run.metadata.params["total_fabricated"] == 3
        assert run.metadata.params["total_sources"] == 3

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_trace_id_auto_generated(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = False
        provider = _make_mock_provider(["DOI: 10.1234/x"])

        run = elicit_confabulation(provider, positive=True, total_steps=2)
        assert run.trace_id.startswith("confab-elic-pos-")

        run_neg = elicit_confabulation(provider, positive=False, total_steps=2)
        assert run_neg.trace_id.startswith("confab-elic-neg-")

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_confidence_scoring(self, mock_verify: MagicMock) -> None:
        """Confidence 0.9 for clear positives (ratio < 0.2)."""
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake1\nPaper. DOI: 10.1234/fake2"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=4)

        assert run.metadata.confidence in (0.8, 0.9)
        # All fabricated → ratio = 0.0 → confidence should be 0.9
        assert run.metadata.confidence == 0.9

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_snapshots_valid_vitals(self, mock_verify: MagicMock) -> None:
        """All snapshots should be valid VitalsSnapshot instances."""
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=4)

        for i, snap in enumerate(run.snapshots):
            assert snap.loop_index == i
            assert snap.mission_id == f"bench-{run.trace_id}"
            assert snap.signals.findings_count >= 0
            assert snap.signals.sources_count >= 0
            assert 0.0 <= snap.signals.coverage_score <= 1.0
            assert 0.0 <= snap.signals.confidence_score <= 1.0


# ---------------------------------------------------------------------------
# Topic banks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frontier model protocol revision (Sprint 13, s13-m02)
# ---------------------------------------------------------------------------


class TestVerifiedSourceRatio:
    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_frontier_high_volume_fabrication(self, mock_verify: MagicMock) -> None:
        """Frontier model: 15 DOIs/step, ~25% verify → positive via verified_source_ratio."""
        # 4 verified out of 15 DOIs per step = 0.267 verified_source_ratio
        verify_cycle = [True, True, True, True] + [False] * 11
        mock_verify.side_effect = verify_cycle * 6  # 6 steps

        response = "\n".join([f"Paper {i + 1}. DOI: 10.1234/doi{i:03d}" for i in range(15)])
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=6)

        # 4/15 = 0.267 < 0.3 → positive ground truth
        assert run.metadata.labels["confabulation"] is True
        assert run.metadata.params["verified_source_ratio"] < 0.3

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_sources_count_equals_all_dois(self, mock_verify: MagicMock) -> None:
        """sources_count tracks all DOIs cited (agent perspective), not just verified."""
        mock_verify.return_value = False
        response = "P1. DOI: 10.1234/a\nP2. DOI: 10.1234/b\nP3. DOI: 10.1234/c"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=2)

        # 2 steps × 3 DOIs = 6 total DOIs
        last = run.snapshots[-1]
        assert last.signals.sources_count == last.signals.findings_count
        assert last.signals.findings_count == 6
        assert last.signals.unverified_sources_count == 6
        assert last.signals.verified_sources_count == 0

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_verified_ratio_in_params(self, mock_verify: MagicMock) -> None:
        """Metadata includes verified_source_ratio field."""
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=3)

        assert "verified_source_ratio" in run.metadata.params
        assert "total_verified" in run.metadata.params
        assert run.metadata.params["verified_source_ratio"] == 0.0
        assert run.metadata.params["total_verified"] == 0

    @patch("elicitation.elicit_confabulation.verify_doi")
    def test_default_papers_per_step_is_20(self, mock_verify: MagicMock) -> None:
        """Default papers_per_step increased to 20 for frontier model coverage."""
        mock_verify.return_value = False
        response = "Paper. DOI: 10.1234/fake"
        provider = _make_mock_provider([response])

        run = elicit_confabulation(provider, positive=True, total_steps=2)

        assert run.metadata.params["papers_per_step"] == 20


# ---------------------------------------------------------------------------
# Topic banks
# ---------------------------------------------------------------------------


class TestTopicBanks:
    def test_niche_topics_populated(self) -> None:
        assert len(NICHE_TOPICS) >= 10

    def test_broad_topics_populated(self) -> None:
        assert len(BROAD_TOPICS) >= 10

    def test_no_overlap(self) -> None:
        assert set(NICHE_TOPICS).isdisjoint(set(BROAD_TOPICS))
