"""Validate the local fin-harness core evaluation dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET = Path(__file__).parent / "datasets" / "fin-harness-core-v1.jsonl"
ROUTING = Path(__file__).parent / "datasets" / "fin-harness-core-v1-routing.jsonl"
EXPECTED_COUNT = 50
EXPECTED_BUCKETS = {
    "grounded_fact": 16,
    "comparison_calculation": 8,
    "synthesis_attribution": 7,
    "clarification_unanswerable": 5,
    "compliance_safety": 6,
    "freshness_dynamic": 4,
    "multiturn_memory": 4,
}


def main() -> None:
    rows = []
    for line_no, line in enumerate(DATASET.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict), f"line {line_no}: object required"
        assert str(row.get("id") or "").strip(), f"line {line_no}: id required"
        inputs = row.get("inputs")
        assert isinstance(inputs, dict), f"line {line_no}: inputs required"
        assert inputs.get("query") or inputs.get("turns"), (
            f"line {line_no}: query or turns required"
        )
        outputs = row.get("outputs")
        assert isinstance(outputs, dict), f"line {line_no}: outputs required"
        assert outputs.get("required_facts") or outputs.get("required_behaviors"), (
            f"line {line_no}: grading target required"
        )
        metadata = row.get("metadata")
        assert isinstance(metadata, dict), f"line {line_no}: metadata required"
        assert metadata.get("bucket") in EXPECTED_BUCKETS, (
            f"line {line_no}: unknown bucket"
        )
        rows.append(row)

    ids = [row["id"] for row in rows]
    assert len(rows) == EXPECTED_COUNT, f"expected {EXPECTED_COUNT}, got {len(rows)}"
    assert len(set(ids)) == len(ids), "duplicate ids"

    routing_rows = [
        json.loads(line)
        for line in ROUTING.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    routing_ids = [row.get("id") for row in routing_rows]
    assert len(routing_rows) == EXPECTED_COUNT, (
        f"expected {EXPECTED_COUNT} routing contracts, got {len(routing_rows)}"
    )
    assert len(set(routing_ids)) == len(routing_ids), "duplicate routing ids"
    assert set(routing_ids) == set(ids), "dataset/routing ids do not match"
    for row in routing_rows:
        assert str(row.get("intent_domain") or "").strip(), (
            f"{row.get('id')}: routing intent_domain required"
        )
        assert row.get("source_policy") in {
            "default",
            "source_locked",
            "fresh_authoritative",
            "no_external_fact",
            "conversation_state",
        }, f"{row.get('id')}: invalid source_policy"

    buckets = Counter(row["metadata"]["bucket"] for row in rows)
    assert dict(buckets) == EXPECTED_BUCKETS, (
        f"bucket mismatch: expected {EXPECTED_BUCKETS}, got {dict(buckets)}"
    )
    print(f"ok: {len(rows)} examples")
    for bucket, count in buckets.items():
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
