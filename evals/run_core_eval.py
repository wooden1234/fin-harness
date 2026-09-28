"""Run selected core cases locally and publish the experiment to LangSmith."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from evals.adapter import HarnessEvalAdapter
from evals.dataset import DATASET_NAME, load_cases, select_cases
from evals.evaluators import (
    evaluate_behavior,
    evaluate_evidence,
    evaluate_facts,
    evaluate_routing,
)
from harness.prompt.assembler import assemble_system, sha256_text


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sdk() -> tuple[Any, Any]:
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate
    except ImportError as exc:  # pragma: no cover - external dependency
        raise SystemExit("langsmith is not installed; run: pip install -r requirements.txt") from exc
    return Client, evaluate


def _as_langsmith_evaluator(function):
    def evaluator(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        return {"results": function(outputs, reference_outputs)}

    evaluator.__name__ = f"langsmith_{function.__name__}"
    return evaluator


def _input_key(inputs: dict[str, Any]) -> str:
    return json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bucket")
    parser.add_argument("--all", action="store_true", dest="all_cases")
    parser.add_argument("--experiment-prefix")
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()
    load_dotenv()

    cases = select_cases(
        load_cases(),
        case_ids=args.case_id,
        bucket=args.bucket,
        limit=args.limit,
        all_cases=args.all_cases,
    )
    if not cases:
        raise SystemExit("no evaluation cases selected")

    Client, evaluate = _sdk()
    client = Client()
    adapter = HarnessEvalAdapter()
    case_id_by_input = {_input_key(case["inputs"]): case["id"] for case in cases}
    wanted = {case["id"] for case in cases}
    remote_by_id = {
        str((example.metadata or {}).get("case_id") or ""): example
        for example in client.list_examples(dataset_name=DATASET_NAME)
        if str((example.metadata or {}).get("case_id") or "") in wanted
    }
    missing = wanted - set(remote_by_id)
    if missing:
        raise SystemExit(
            f"LangSmith dataset is missing cases {sorted(missing)}; "
            "run: python -m evals.upload_dataset"
        )
    remote_examples = [remote_by_id[case["id"]] for case in cases]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(
            adapter.run_case(inputs, case_id=case_id_by_input.get(_input_key(inputs), ""))
        )

    commit = _git_commit()
    model = os.getenv("DEEPSEEK_MODEL", "unknown")
    prefix = args.experiment_prefix or (
        f"{DATASET_NAME}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{commit}"
    )
    metadata = {
        "dataset": DATASET_NAME,
        "git_commit": commit,
        "prompt_sha256": sha256_text(assemble_system()),
        "model": model,
        "app_env": os.getenv("APP_ENV", "unknown"),
        "case_count": len(cases),
    }
    results = evaluate(
        target,
        data=remote_examples,
        evaluators=[
            _as_langsmith_evaluator(evaluate_facts),
            _as_langsmith_evaluator(evaluate_evidence),
            _as_langsmith_evaluator(evaluate_routing),
            _as_langsmith_evaluator(evaluate_behavior),
        ],
        experiment_prefix=prefix,
        description="fin-harness core offline evaluation",
        metadata=metadata,
        max_concurrency=max(1, args.max_concurrency),
        client=client,
        error_handling="log",
    )
    print(results)


if __name__ == "__main__":
    main()
