"""Tests for delayed-onset confab corpus expansion utilities."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.expand_delayed_confab_corpus import (
    TRACE_PREFIXES,
    build_delayed_confab_entries,
    cleanup_existing_delayed_traces,
    merge_manifest,
)


def test_build_delayed_confab_entries_hits_required_mix(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus" / "v1"
    entries = build_delayed_confab_entries(corpus_root=corpus_root)

    positives = [entry for entry in entries if entry["labels"]["confabulation"]]
    negatives = [entry for entry in entries if not entry["labels"]["confabulation"]]

    assert len(positives) == 36
    assert len(negatives) == 9
    assert all(entry["metadata"]["confidence"] == 1.0 for entry in entries)
    assert {entry["metadata"]["params"]["pattern"] for entry in positives} == {
        "delayed_sharp",
        "delayed_gradual",
        "delayed_oscillating",
    }
    assert {entry["metadata"]["params"]["pattern"] for entry in negatives} == {
        "delayed_sharp",
        "delayed_gradual",
        "delayed_oscillating",
    }
    assert all(entry["path"].startswith("traces/confabulation/") for entry in entries)
    assert len(list((corpus_root / "traces" / "confabulation" / "positive").glob("*.json"))) == 36
    assert len(list((corpus_root / "traces" / "confabulation" / "negative").glob("*.json"))) == 9


def test_cleanup_existing_delayed_traces_only_removes_target_prefixes(tmp_path: Path) -> None:
    positive_dir = tmp_path / "corpus" / "v1" / "traces" / "confabulation" / "positive"
    positive_dir.mkdir(parents=True, exist_ok=True)
    delayed = positive_dir / "confab-syn-delayed-001.json"
    unrelated = positive_dir / "confab-syn-healthy-001.json"
    delayed.write_text("[]\n")
    unrelated.write_text("[]\n")

    cleanup_existing_delayed_traces(corpus_root=tmp_path / "corpus" / "v1")

    assert not delayed.exists()
    assert unrelated.exists()


def test_merge_manifest_replaces_existing_delayed_entries(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    existing = [
        {"trace_id": "confab-syn-delayed-001", "path": "old.json", "labels": {}, "metadata": {}},
        {"trace_id": "loop-syn-001", "path": "loop.json", "labels": {}, "metadata": {}},
    ]
    manifest_path.write_text(json.dumps(existing, indent=2) + "\n")

    merged = merge_manifest(
        entries=[
            {
                "trace_id": "confab-syn-delayed-001",
                "path": "new.json",
                "labels": {"confabulation": True},
                "metadata": {"params": {"pattern": "delayed_sharp"}},
            },
            {
                "trace_id": "confab-syn-delayed-neg-001",
                "path": "new-neg.json",
                "labels": {"confabulation": False},
                "metadata": {"params": {"pattern": "delayed_sharp"}},
            },
        ],
        manifest_path=manifest_path,
    )

    trace_ids = [entry["trace_id"] for entry in merged]
    assert "loop-syn-001" in trace_ids
    assert trace_ids.count("confab-syn-delayed-001") == 1
    assert any(entry["trace_id"].startswith(TRACE_PREFIXES) for entry in merged)
