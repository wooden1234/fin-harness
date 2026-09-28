"""Idempotently upload the local core dataset to LangSmith."""

from __future__ import annotations

import argparse
from typing import Any

from dotenv import load_dotenv

from evals.dataset import DATASET_NAME, load_cases


def _client() -> Any:
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise SystemExit("langsmith is not installed; run: pip install -r requirements.txt") from exc
    return Client()


def upload_dataset(*, dataset_name: str = DATASET_NAME, client: Any | None = None) -> dict[str, int | str]:
    client = client or _client()
    cases = load_cases()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception as exc:  # LangSmith SDK has no stable typed NotFound across versions
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            has_dataset = getattr(client, "has_dataset", None)
            if not callable(has_dataset) or has_dataset(dataset_name=dataset_name):
                raise
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="fin-harness 端到端企业级核心离线评测集",
        )

    existing: set[str] = set()
    for example in client.list_examples(dataset_id=dataset.id):
        metadata = dict(getattr(example, "metadata", None) or {})
        case_id = str(metadata.get("case_id") or "")
        if case_id:
            existing.add(case_id)

    pending = [case for case in cases if case["id"] not in existing]
    if pending:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": case["inputs"],
                    "outputs": case["reference_outputs"],
                    "metadata": case["metadata"],
                }
                for case in pending
            ],
        )
    return {
        "dataset_name": dataset_name,
        "created": len(pending),
        "skipped": len(cases) - len(pending),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()
    load_dotenv()
    print(upload_dataset(dataset_name=args.dataset_name))


if __name__ == "__main__":
    main()
