"""Tests for per-framework profile evaluation support (s09-m01)."""

from __future__ import annotations

import pytest
from agent_vitals import VitalsConfig

from evaluator.runner import run_evaluation


class TestProfileConfig:
    """Test that framework profiles correctly override VitalsConfig."""

    def test_for_framework_returns_modified_config(self) -> None:
        base = VitalsConfig.from_yaml()
        lg = base.for_framework("langgraph")
        # langgraph profile overrides loop_consecutive_pct to 0.4
        assert lg.loop_consecutive_pct == pytest.approx(0.4)
        # base should be unchanged
        assert base.loop_consecutive_pct == pytest.approx(0.5)

    def test_for_framework_crewai(self) -> None:
        base = VitalsConfig.from_yaml()
        cr = base.for_framework("crewai")
        assert cr.burn_rate_multiplier == pytest.approx(3.0)
        assert cr.token_scale_factor == pytest.approx(0.7)

    def test_for_framework_dspy(self) -> None:
        base = VitalsConfig.from_yaml()
        dspy = base.for_framework("dspy")
        assert dspy.stuck_dm_threshold == pytest.approx(0.1)

    def test_unknown_framework_returns_base(self) -> None:
        base = VitalsConfig.from_yaml()
        same = base.for_framework("unknown_framework")
        assert same.loop_consecutive_pct == base.loop_consecutive_pct
        assert same.burn_rate_multiplier == base.burn_rate_multiplier


class TestProfileRunEvaluation:
    """Test that run_evaluation correctly threads profile through."""

    def test_evaluation_result_has_profile(self) -> None:
        result = run_evaluation(corpus_version="v1", profile="langgraph")
        assert result.profile == "langgraph"

    def test_evaluation_result_default_profile_is_none(self) -> None:
        result = run_evaluation(corpus_version="v1")
        assert result.profile is None

    def test_profile_filters_framework_traces(self) -> None:
        """When profile is set, traces with a different framework field are excluded."""
        # Run with no profile — gets all traces
        all_result = run_evaluation(corpus_version="v1")
        # Run with a profile — gets only matching (or untagged) traces
        profiled_result = run_evaluation(corpus_version="v1", profile="langgraph")
        # Profiled should have <= all traces (most v1 traces are untagged,
        # so they pass the filter; but any tagged with a different framework
        # should be excluded)
        assert profiled_result.trace_count <= all_result.trace_count
        assert profiled_result.profile == "langgraph"


class TestProfileReporter:
    """Test that reports include profile information."""

    def test_report_includes_profile_label(self) -> None:
        from evaluator.reporter import report_results

        result = run_evaluation(corpus_version="v1", profile="crewai")
        report = report_results(result, save=False)
        assert "**Profile:** crewai" in report

    def test_report_default_profile_label(self) -> None:
        from evaluator.reporter import report_results

        result = run_evaluation(corpus_version="v1")
        report = report_results(result, save=False)
        assert "**Profile:** default" in report
