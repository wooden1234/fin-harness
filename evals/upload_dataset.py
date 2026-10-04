"""Idempotently upload the local core dataset to LangSmith."""

from __future__ import annotations

import argparse
from typing import Any

from dotenv import load_dotenv

from evals.dataset import (
    DATASET_NAME,
    PDF_ANSWER_DATASET_NAME,
    PDF_ANSWER_V2_DATASET_NAME,
    load_cases,
    load_pdf_answer_cases,
    load_pdf_answer_v2_cases,
)


def _client() -> Any:
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise SystemExit("langsmith is not installed; run: pip install -r requirements.txt") from exc
    return Client()


def _read_or_create_dataset(client: Any, dataset_name: str, description: str) -> Any:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception as exc:  # LangSmith SDK has no stable typed NotFound across versions
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            has_dataset = getattr(client, "has_dataset", None)
            if not callable(has_dataset) or has_dataset(dataset_name=dataset_name):
                raise
        return client.create_dataset(dataset_name=dataset_name, description=description)


def _example_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "inputs": case["inputs"],
        "outputs": case["reference_outputs"],
        "metadata": case["metadata"],
    }


def _upload_cases(
    cases: list[dict[str, Any]],
    *,
    dataset_name: str,
    description: str,
    client: Any | None = None,
    update_existing: bool = False,
) -> dict[str, int | str]:
    client = client or _client()
    dataset = _read_or_create_dataset(client, dataset_name, description)

    existing: dict[str, Any] = {}
    for example in client.list_examples(dataset_id=dataset.id):
        metadata = dict(getattr(example, "metadata", None) or {})
        case_id = str(metadata.get("case_id") or "")
        if case_id:
            existing[case_id] = example

    pending = [case for case in cases if case["id"] not in existing]
    updated = 0
    if pending:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[_example_payload(case) for case in pending],
        )
    if update_existing:
        for case in cases:
            example = existing.get(case["id"])
            if example is None:
                continue
            client.update_example(
                example_id=example.id,
                inputs=case["inputs"],
                outputs=case["reference_outputs"],
                metadata=case["metadata"],
            )
            updated += 1
    return {
        "dataset_name": dataset_name,
        "dataset_id": str(getattr(dataset, "id", "")),
        "url": str(getattr(dataset, "url", "") or ""),
        "created": len(pending),
        "updated": updated,
        "skipped": 0 if update_existing else len(cases) - len(pending),
    }


def upload_dataset(*, dataset_name: str = DATASET_NAME, client: Any | None = None) -> dict[str, int | str]:
    return _upload_cases(
        load_cases(),
        dataset_name=dataset_name,
        description="fin-harness 端到端企业级核心离线评测集",
        client=client,
    )


def upload_pdf_answer_dataset(
    *,
    dataset_name: str = PDF_ANSWER_DATASET_NAME,
    client: Any | None = None,
) -> dict[str, int | str]:
    return _upload_cases(
        load_pdf_answer_cases(),
        dataset_name=dataset_name,
        description="PDF Agent 回答准确性与忠实度评测集",
        client=client,
        update_existing=True,
    )


def upload_pdf_answer_v2_dataset(
    *,
    dataset_name: str = PDF_ANSWER_V2_DATASET_NAME,
    client: Any | None = None,
) -> dict[str, int | str]:
    return _upload_cases(
        load_pdf_answer_v2_cases(),
        dataset_name=dataset_name,
        description="PDF Agent 回答评测 3 条 demo 集",
        client=client,
        update_existing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="")
    parser.add_argument(
        "--source",
        choices=("core", "pdf-answer", "pdf-answer-v2"),
        default="core",
        help="core 上传端到端核心集；pdf-answer 上传 50 条回答集；pdf-answer-v2 上传 3 条 demo 集",
    )
    args = parser.parse_args()
    load_dotenv()
    if args.source == "pdf-answer":
        print(upload_pdf_answer_dataset(dataset_name=args.dataset_name or PDF_ANSWER_DATASET_NAME))
        return
    if args.source == "pdf-answer-v2":
        print(upload_pdf_answer_v2_dataset(dataset_name=args.dataset_name or PDF_ANSWER_V2_DATASET_NAME))
        return
    print(upload_dataset(dataset_name=args.dataset_name or DATASET_NAME))


if __name__ == "__main__":
    main()
