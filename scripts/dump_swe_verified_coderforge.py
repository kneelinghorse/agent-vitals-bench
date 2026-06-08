#!/usr/bin/env python3
"""
Dump raw SWE-bench Verified + CoderForge trajectories to a single JSONL.

This stages the raw trajectories locally so M02's VitalsSnapshot adapter and
M03's evaluation run can work offline from a committed artifact instead of
re-pulling from HuggingFace on every run.

Sources (both already in the HF cache from metrics_and_protocols'
TR-20260418-001 dataset-adapters mission):
  - princeton-nlp/SWE-bench_Verified (issue metadata, split=test)
  - togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories
    (trajectory messages + binary reward, split=train, config=trajectory)

Each output record joins a CoderForge trajectory with its matching Verified
issue by instance_id. Trajectories whose instance_id is absent from Verified
are dropped (matches the upstream adapter's behavior).

Run with the metrics_and_protocols pivot_probe_env (has `datasets` installed):
  /Users/mac-studio/projects/unified-system-state/metrics_and_protocols/tmp/pivot_probe_env/bin/python \\
      scripts/dump_swe_verified_coderforge.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

ISSUE_DATASET = "princeton-nlp/SWE-bench_Verified"
TRAJECTORY_DATASET = (
    "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"
)
TRAJECTORY_CONFIG = "trajectory"

OUT_DIR = Path(__file__).resolve().parent.parent / "corpus" / "academic" / "swe_verified_coderforge"
OUT_PATH = OUT_DIR / "trajectories.jsonl"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    verified = load_dataset(ISSUE_DATASET, split="test")
    verified_by_id = {row["instance_id"]: row for row in verified}
    trajectories = load_dataset(TRAJECTORY_DATASET, TRAJECTORY_CONFIG, split="train")

    written = 0
    missing = 0
    non_binary_rewards = 0
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for index, trajectory in enumerate(trajectories):
            ds_payload = trajectory.get("ds")
            if isinstance(ds_payload, str):
                ds_payload = json.loads(ds_payload)
            if not isinstance(ds_payload, dict):
                ds_payload = {}

            instance_id = ds_payload.get("instance_id")
            if instance_id not in verified_by_id:
                missing += 1
                continue

            reward = trajectory.get("reward")
            if reward not in {0.0, 1.0}:
                non_binary_rewards += 1

            record = {
                "source_row_index": index,
                "trajectory_id": trajectory.get("trajectory_id")
                or f"{instance_id}:run{trajectory.get('run_id', 1)}",
                "instance_id": instance_id,
                "run_id": trajectory.get("run_id"),
                "reward": reward,
                "messages": trajectory.get("messages"),
                "output_patch": trajectory.get("output_patch"),
                "verified_issue": {
                    "problem_statement": verified_by_id[instance_id].get("problem_statement"),
                    "hints_text": verified_by_id[instance_id].get("hints_text"),
                    "repo": verified_by_id[instance_id].get("repo"),
                    "base_commit": verified_by_id[instance_id].get("base_commit"),
                    "test_patch": verified_by_id[instance_id].get("test_patch"),
                    "patch": verified_by_id[instance_id].get("patch"),
                },
            }
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")
            written += 1

    sha = hashlib.sha256()
    with OUT_PATH.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)

    print(f"wrote {written} trajectories to {OUT_PATH}")
    print(f"  missing_from_verified: {missing}")
    print(f"  non_binary_rewards: {non_binary_rewards}")
    print(f"  sha256: {sha.hexdigest()}")


if __name__ == "__main__":
    main()
