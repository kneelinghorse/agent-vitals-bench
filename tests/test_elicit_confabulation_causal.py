"""Tests for causal-aware confabulation elicitation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from elicitation.elicit_confabulation_causal import (
    CausalConfabRun,
    elicit_confabulation_causal,
    extract_supported_findings,
)
from elicitation.providers import GenerateResult, Provider


def _make_mock_provider(
    responses: list[str], name: str = "test-provider", model: str = "test-model"
) -> Provider:
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
            prompt_tokens=120,
            completion_tokens=180,
            total_tokens=300,
            model=model,
            provider=name,
        )

    mock.generate = _generate
    return mock


class TestExtractSupportedFindings:
    def test_parses_structured_findings(self) -> None:
        text = (
            "FINDING: MXene coatings improve corrosion resistance.\n"
            "DOI: 10.1234/example001\n"
            "FINDING: Zinc ion release can remain cytocompatible.\n"
            "DOI: 10.1234/example002\n"
        )

        findings = extract_supported_findings(text)

        assert len(findings) == 2
        assert findings[0].doi == "10.1234/example001"
        assert findings[1].doi == "10.1234/example002"

    def test_falls_back_to_unstructured_response(self) -> None:
        findings = extract_supported_findings("A claim without a DOI.")

        assert len(findings) == 1
        assert findings[0].doi is None


class TestCausalConfabulationElicitation:
    @patch("elicitation.elicit_confabulation_causal.verify_doi")
    def test_positive_trace_decouples_findings_from_verified_sources(
        self, mock_verify: MagicMock
    ) -> None:
        mock_verify.return_value = False
        response = (
            "FINDING: Claim one.\nDOI: 10.1234/fake001\n"
            "FINDING: Claim two.\nDOI: 10.1234/fake002\n"
            "FINDING: Claim three.\nDOI: 10.1234/fake003\n"
        )
        provider = _make_mock_provider([response])

        run = elicit_confabulation_causal(
            provider,
            positive=True,
            total_steps=4,
            findings_per_step=3,
            trace_id="causal-pos-001",
        )

        assert isinstance(run, CausalConfabRun)
        assert run.metadata.labels["confabulation"] is True
        assert run.metadata.params["distinct_verified_sources"] == 0
        assert run.metadata.params["unsupported_findings"] == 12
        assert run.snapshots[-1].signals.findings_count == 12
        assert run.snapshots[-1].signals.sources_count == 0
        assert run.snapshots[-1].source_finding_ratio == 0.0

    @patch("elicitation.elicit_confabulation_causal.verify_doi")
    def test_negative_trace_keeps_support_ratio_healthy(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = True
        provider = _make_mock_provider(
            [
                (
                    "FINDING: Claim one.\nDOI: 10.1234/real001\n"
                    "FINDING: Claim two.\nDOI: 10.1234/real002\n"
                    "FINDING: Claim three.\nDOI: 10.1234/real003\n"
                ),
                (
                    "FINDING: Claim four.\nDOI: 10.1234/real004\n"
                    "FINDING: Claim five.\nDOI: 10.1234/real005\n"
                    "FINDING: Claim six.\nDOI: 10.1234/real006\n"
                ),
                (
                    "FINDING: Claim seven.\nDOI: 10.1234/real007\n"
                    "FINDING: Claim eight.\nDOI: 10.1234/real008\n"
                    "FINDING: Claim nine.\nDOI: 10.1234/real009\n"
                ),
            ]
        )

        run = elicit_confabulation_causal(
            provider,
            positive=False,
            total_steps=3,
            findings_per_step=3,
            trace_id="causal-neg-001",
        )

        assert run.metadata.labels["confabulation"] is False
        assert run.metadata.params["distinct_verified_sources"] == 9
        assert run.metadata.params["final_supported_ratio"] == 1.0
        assert run.snapshots[-1].signals.sources_count == 9
        assert run.snapshots[-1].signals.findings_count == 9

    @patch("elicitation.elicit_confabulation_causal.verify_doi")
    def test_metadata_records_protocol_and_provider(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = False
        response = "FINDING: Claim one.\nDOI: 10.1234/fake001\n"
        provider = _make_mock_provider([response], name="deepseek", model="deepseek-chat")

        run = elicit_confabulation_causal(provider, positive=True, total_steps=2)

        assert run.metadata.params["protocol"] == "causal_support"
        assert run.metadata.params["provider"] == "deepseek"
        assert run.metadata.params["model"] == "deepseek-chat"
