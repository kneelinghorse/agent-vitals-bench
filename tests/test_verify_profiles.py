"""Tests for scripts/verify_profiles.py — the v1.14.0-API per-framework gate.

These tests lock the bench-side expected contract for which framework profiles
exist and how each one diverges from pure ``VitalsConfig()`` defaults. They use
the public ``VitalsConfig`` profile introspection API exclusively (no
``dataclasses.fields()`` introspection, no private attribute access).

Any change to ``EXPECTED_DIVERGENCES`` here is a coordinated contract change
with agent-vitals and must be paired with an updated thresholds.yaml release.
"""

from __future__ import annotations

import pytest
from agent_vitals import UnknownProfileError, VitalsConfig

from scripts.verify_profiles import EXPECTED_DIVERGENCES, verify_profiles


class TestProfileIntrospectionAPI:
    """Direct exercises of the v1.14.0 public surface."""

    def test_yaml_is_loaded(self) -> None:
        assert VitalsConfig.is_yaml_loaded() is True

    def test_assert_profiles_loaded_passes(self) -> None:
        # Must not raise against a normally installed wheel.
        VitalsConfig.assert_profiles_loaded()

    def test_list_profiles_matches_expected_set(self) -> None:
        assert set(VitalsConfig.list_profiles()) == set(EXPECTED_DIVERGENCES)

    def test_instance_profiles_matches_classmethod(self) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert set(cfg.profiles()) == set(VitalsConfig.list_profiles())

    def test_unknown_profile_raises_typed_error(self) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        with pytest.raises(UnknownProfileError):
            cfg.profile_diff("__bench_unknown_sentinel__")


class TestProfileDivergences:
    """Lock the per-profile divergence contract."""

    @pytest.mark.parametrize("framework", sorted(EXPECTED_DIVERGENCES))
    def test_profile_diff_matches_expected(self, framework: str) -> None:
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        diff = cfg.profile_diff(framework)
        actual = {d.field: (d.default, d.override) for d in diff.values()}
        assert actual == EXPECTED_DIVERGENCES[framework]

    @pytest.mark.parametrize("framework", sorted(EXPECTED_DIVERGENCES))
    def test_profile_diff_is_nonempty(self, framework: str) -> None:
        # A profile with zero overrides would be a silent regression
        # (e.g. the v1.13.0 "yaml not bundled" packaging bug).
        cfg = VitalsConfig.from_yaml(allow_env_override=False)
        assert cfg.profile_diff(framework), (
            f"profile {framework!r} has no overrides — silent YAML drift"
        )


class TestVerifyProfilesScript:
    """End-to-end gate: verify_profiles() returns no failures."""

    def test_verify_profiles_passes(self) -> None:
        failures = verify_profiles()
        assert failures == [], f"verify_profiles failures: {failures}"
