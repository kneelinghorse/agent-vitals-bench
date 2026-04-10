"""Verify the agent-vitals framework profile bundle via the public introspection API.

This is the bench-side gate against the v1.13.0 packaging regression class:
the wheel ships, the default config still works, but framework profiles are
silently absent or silently equal to defaults. v1.14.0 added a stable public
``VitalsConfig`` profile introspection API so this gate no longer needs to
poke at private dataclass internals — agent-vitals is now the single source
of truth for which profiles exist and how they diverge from defaults.

Exits non-zero on any of:
  * thresholds.yaml not bundled / not loaded
  * any expected profile missing
  * any profile present but with no overrides (silent default)
  * any profile divergence diverging from the bench-side expected contract

Run from the bench main venv against an installed agent-vitals wheel:
  .venv/bin/python scripts/verify_profiles.py
"""

from __future__ import annotations

from agent_vitals import (
    ProfileFieldDiff,
    UnknownProfileError,
    VitalsConfig,
)
from agent_vitals.exceptions import ConfigurationError

# Bench-side expected contract: profile name -> {field: (default, override)}.
# Source: agent-vitals thresholds.yaml as validated in the v1.13.1 PyPI
# verification report (reports/eval-agent-vitals-v1.13.1-pypi-verification.md).
# Updates to this contract require a coordinated PR with agent-vitals.
EXPECTED_DIVERGENCES: dict[str, dict[str, tuple[object, object]]] = {
    "crewai": {
        # burn_rate_multiplier override removed in agent-vitals v1.14.1 — it
        # was triggering 34 stuck FPs on short runaway-positive traces via an
        # implicit `burn_rate_anomaly` arbitration-plus-filter side channel.
        # crewai now inherits the default burn_rate_multiplier=2.5.
        "token_scale_factor": (1.0, 0.7),
    },
    "dspy": {
        "loop_consecutive_pct": (0.5, 0.7),
        "stuck_dm_threshold": (0.15, 0.1),
        "workflow_stuck_enabled": ("research-only", "none"),
    },
    "langgraph": {
        "loop_consecutive_pct": (0.5, 0.4),
    },
}


def _format_diff(diff: dict[str, ProfileFieldDiff]) -> str:
    return ", ".join(f"{d.field}: {d.default!r}→{d.override!r}" for d in diff.values())


def verify_profiles() -> list[str]:
    """Run the full per-framework verification gate.

    Returns a list of failure messages. An empty list means PASS.
    """
    failures: list[str] = []

    # 1. Hard gate: yaml must be loaded and contain at least one profile.
    try:
        VitalsConfig.assert_profiles_loaded()
    except ConfigurationError as exc:
        failures.append(f"assert_profiles_loaded() raised: {exc}")
        return failures  # nothing else makes sense if the yaml is missing

    # 2. Every expected profile must be present.
    known = set(VitalsConfig.list_profiles())
    expected = set(EXPECTED_DIVERGENCES)
    missing = expected - known
    if missing:
        failures.append(f"missing profiles: {sorted(missing)} (known: {sorted(known)})")

    cfg = VitalsConfig.from_yaml(allow_env_override=False)
    cfg_profiles = set(cfg.profiles())
    if cfg_profiles != known:
        failures.append(
            f"VitalsConfig.list_profiles() {sorted(known)} != cfg.profiles() {sorted(cfg_profiles)}"
        )

    # 3. Per-profile divergence must match the expected contract.
    for fw, expected_overrides in EXPECTED_DIVERGENCES.items():
        if fw not in known:
            continue  # already reported above
        try:
            diff = cfg.profile_diff(fw)
        except UnknownProfileError as exc:
            failures.append(f"profile_diff({fw!r}) raised UnknownProfileError: {exc}")
            continue

        if not diff:
            failures.append(f"profile {fw!r} has no overrides — possible silent YAML drift")
            continue

        actual = {d.field: (d.default, d.override) for d in diff.values()}
        if actual != expected_overrides:
            failures.append(
                f"profile {fw!r} divergence mismatch: expected {expected_overrides}, got {actual}"
            )

    # 4. Unknown profile must raise the typed error.
    try:
        cfg.profile_diff("__bench_unknown_sentinel__")
    except UnknownProfileError:
        pass
    except Exception as exc:  # noqa: BLE001 — we want to flag any other error type
        failures.append(
            f"profile_diff('__bench_unknown_sentinel__') raised {type(exc).__name__}, "
            f"expected UnknownProfileError"
        )
    else:
        failures.append(
            "profile_diff('__bench_unknown_sentinel__') did not raise UnknownProfileError"
        )

    return failures


def main() -> int:
    av_version = __import__("agent_vitals").__version__
    print(f"agent-vitals version: {av_version}")
    print(f"thresholds.yaml path: {VitalsConfig.thresholds_yaml_path()}")
    print(f"is_yaml_loaded:       {VitalsConfig.is_yaml_loaded()}")
    print(f"list_profiles:        {VitalsConfig.list_profiles()}")
    print()

    cfg = VitalsConfig.from_yaml(allow_env_override=False)
    for fw in sorted(cfg.profiles()):
        try:
            diff = cfg.profile_diff(fw)
        except UnknownProfileError as exc:
            print(f"  {fw}: ERROR {exc}")
            continue
        print(f"  {fw}: {_format_diff(diff)}")
    print()

    failures = verify_profiles()
    if failures:
        print("PROFILE VERIFICATION: FAIL")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("PROFILE VERIFICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
