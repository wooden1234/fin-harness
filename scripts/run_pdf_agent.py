"""只调用 PDF Agent 图，并把节点树上报到 LangSmith。

不经过 Finance Agent、规划器或 Web。需要 .env 里 LANGSMITH_TRACING=true。

用法：
  python scripts/run_pdf_agent.py --query "寒武纪2024年营业收入是多少"
  python scripts/run_pdf_agent.py --cases retrieval/eval/answer_faithfulness.jsonl --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env")

from agents.finance_agent.pdf_agent import get_pdf_agent_graph  # noqa: E402


def _queries(args: argparse.Namespace) -> list[str]:
    if args.query:
        return [args.query]
    cases = []
    for line in Path(args.cases).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = str(row.get("query") or "").strip()
        if query:
            cases.append(query)
    if args.limit:
        cases = cases[: max(int(args.limit), 1)]
    return cases


async def _run(query: str) -> None:
    graph = get_pdf_agent_graph()
    result = await graph.ainvoke(
        {
            "original_query": query,
            "query": query,
            "rewrite_count": 0,
            "messages": [],
        },
        config={
            "run_name": "pdf_agent",
            "tags": ["pdf_agent"],
            "metadata": {"entrypoint": "pdf_agent_only"},
        },
    )
    answer = str(result.get("answer") or "").replace("\n", " ")
    print(
        f"route={result.get('evidence_route') or '-'} "
        f"rewrite={result.get('rewrite_strategy') or '-'} "
        f"answer={answer[:180]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="只跑 PDF Agent 图")
    parser.add_argument("--query", default="")
    parser.add_argument("--cases", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.query and not args.cases:
        parser.error("需要 --query 或 --cases")
    queries = _queries(args)
    if not queries:
        raise SystemExit("没有可运行的问题")
    for query in queries:
        print(f"\n# {query}")
        asyncio.run(_run(query))


if __name__ == "__main__":
    main()
