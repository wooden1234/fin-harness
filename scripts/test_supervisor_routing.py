"""Supervisor 结构化路由联调（Week 3 Day 2）。

用法:
    source .venv/bin/activate
    python scripts/test_supervisor_routing.py
    python scripts/test_supervisor_routing.py "港股通有哪些特殊规则？"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.agents.supervisor import analyze_and_route_query, route_query  # noqa: E402

SAMPLE_QUERIES = [
    "什么是 T+1 交易制度？",
    "我的可用资金还有多少？",
    "今天上海天气怎么样？",
    "我被人骗转账了 50 万怎么办？",
]


async def run_one(query: str) -> None:
    state = {"messages": [HumanMessage(content=query)]}
    update = await analyze_and_route_query(state, {})
    merged = {**state, **update}
    target = route_query(merged)
    print(f"\nQ: {query}")
    print(f"  route       = {update.get('route')}")
    print(f"  risk_level  = {update.get('risk_level')}")
    print(f"  logic       = {update.get('logic')}")
    print(f"  next_node   = {target}")


async def main() -> None:
    queries = sys.argv[1:] or SAMPLE_QUERIES
    for q in queries:
        await run_one(q)


if __name__ == "__main__":
    asyncio.run(main())
