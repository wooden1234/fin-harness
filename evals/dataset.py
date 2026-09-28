"""Load and merge the versioned core dataset and routing contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "datasets" / "fin-harness-core-v1.jsonl"
ROUTING_PATH = ROOT / "datasets" / "fin-harness-core-v1-routing.jsonl"
DATASET_NAME = "fin-harness-core-v1"

CALIBRATION_IDS = (
    "core-001",
    "core-005",
    "core-011",
    "core-014",
    "core-017",
    "core-025",
    "core-032",
    "core-037",
    "core-043",
    "core-047",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: JSON object required")
        rows.append(value)
    return rows


def load_cases() -> list[dict[str, Any]]:
    cases = _read_jsonl(DATASET_PATH)
    contracts = _read_jsonl(ROUTING_PATH)
    by_id = {str(item.get("id") or ""): item for item in contracts}
    if len(by_id) != len(contracts):
        raise ValueError("duplicate routing contract id")

    merged: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id not in by_id:
            raise ValueError(f"missing routing contract for {case_id or '<empty>'}")
        outputs = dict(case.get("outputs") or {})
        outputs["routing_contract"] = {
            key: value for key, value in by_id[case_id].items() if key != "id"
        }
        metadata = {"case_id": case_id, **dict(case.get("metadata") or {})}
        merged.append(
            {
                "id": case_id,
                "inputs": dict(case.get("inputs") or {}),
                "reference_outputs": outputs,
                "metadata": metadata,
            }
        )

    case_ids = {item["id"] for item in merged}
    extras = set(by_id) - case_ids
    if extras:
        raise ValueError(f"routing contracts without cases: {sorted(extras)}")
    return merged


def select_cases(
    cases: Iterable[dict[str, Any]],
    *,
    case_ids: Iterable[str] = (),
    bucket: str | None = None,
    limit: int | None = None,
    all_cases: bool = False,
) -> list[dict[str, Any]]:
    rows = list(cases)
    wanted = {str(item) for item in case_ids if str(item)}
    if wanted:
        selected = [row for row in rows if row["id"] in wanted]
        missing = wanted - {row["id"] for row in selected}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    elif bucket:
        selected = [row for row in rows if row["metadata"].get("bucket") == bucket]
    elif all_cases:
        selected = rows
    else:
        selected = [row for row in rows if row["id"] in CALIBRATION_IDS]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


__all__ = [
    "CALIBRATION_IDS",
    "DATASET_NAME",
    "load_cases",
    "select_cases",
]
