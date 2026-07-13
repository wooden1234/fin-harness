"""运行 knowledge/eval/planner_eval.jsonl，评测 finance_agent planner 拆分准确率。

用法:
  python scripts/run_planner_eval.py
  python scripts/run_planner_eval.py --min-type-accuracy 0.8
  python scripts/run_planner_eval.py --dry-run   # 只校验评测集格式，不调 LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.agents.finance_agent.planner.eval_cases import (  # noqa: E402
    DEFAULT_EVAL_PATH,
    load_eval_cases,
    score_case,
    summarize,
)

DEFAULT_MIN_TYPE_ACCURACY = 0.8
DEFAULT_MIN_EMPTY_ACCURACY = 0.8


async def _run_live(cases: list[dict]) -> list[dict]:
    from langchain_core.messages import HumanMessage

    from app.agents.finance_agent.planner.node import supervisor_node

    rows: list[dict] = []
    for case in cases:
        out = await supervisor_node(
            {"messages": [HumanMessage(content=case["query"])]},
            {},
        )
        tasks = list(out.get("sub_tasks") or [])
        actual_types = [str(t.type) for t in tasks]
        row = score_case(case, actual_types)
        row["steps"] = out.get("steps")
        rows.append(row)
        mark = "OK" if row["type_ok"] and row["empty_ok"] else "FAIL"
        print(
            f"[{mark}] {case['id']}: expected={case['expected_types']} actual={actual_types}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval finance_agent planner decomposition")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--dry-run", action="store_true", help="只校验评测集格式")
    parser.add_argument(
        "--min-type-accuracy",
        type=float,
        default=DEFAULT_MIN_TYPE_ACCURACY,
        help="type 准确率阈值，低于则 exit 1",
    )
    parser.add_argument(
        "--min-empty-accuracy",
        type=float,
        default=DEFAULT_MIN_EMPTY_ACCURACY,
        help="空计划准确率阈值，低于则 exit 1",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "knowledge" / "eval" / "planner_eval_summary.json",
    )
    args = parser.parse_args()

    cases = load_eval_cases(args.eval)
    print(f"loaded {len(cases)} cases from {args.eval}")

    if args.dry_run:
        print("dry-run ok")
        return 0

    rows = asyncio.run(_run_live(cases))
    summary = summarize(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "summary: "
        f"type_accuracy={summary['type_accuracy']:.3f} "
        f"empty_accuracy={summary['empty_accuracy']:.3f} "
        f"count_accuracy={summary['count_accuracy']:.3f}"
    )
    print(
        "triple-split: "
        f"n={summary['triple_total']} "
        f"type_accuracy={summary['triple_type_accuracy']:.3f} "
        f"count_accuracy={summary['triple_count_accuracy']:.3f}"
    )
    print(f"wrote {args.summary_out}")

    failed = False
    if summary["type_accuracy"] < args.min_type_accuracy:
        print(
            f"FAIL type_accuracy {summary['type_accuracy']:.3f} "
            f"< min {args.min_type_accuracy:.3f}"
        )
        failed = True
    if summary["empty_accuracy"] < args.min_empty_accuracy:
        print(
            f"FAIL empty_accuracy {summary['empty_accuracy']:.3f} "
            f"< min {args.min_empty_accuracy:.3f}"
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
