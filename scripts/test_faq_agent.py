"""FAQ 节点联调：Retriever → context → LLM。

用法:
    source .venv/bin/activate
    python scripts/test_faq_agent.py
    python scripts/test_faq_agent.py "什么是 T+1 交易制度？"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env")

from agents.finance_agent.faq_agent import faq_agent  # noqa: E402

SAMPLE_QUERIES = [
    "什么是 T+1 交易制度？",
    "集合竞价的成交原则是什么？",
    "今天北京天气怎么样？",  # 预期拒答或弱相关
]


async def run_one(query: str) -> None:
    state = {"messages": [HumanMessage(content=query)]}
    out = await faq_agent(state, {})
    answer = out["messages"][0].content
    citations = out.get("citations", [])
    print(f"\nQ: {query}")
    print(f"--- answer ---\n{answer}")
    print(f"--- citations ({len(citations)}) ---")
    for i, c in enumerate(citations, start=1):
        print(f"  [{i}] {c.get('source')} | {c.get('snippet', '')[:80]}...")


async def main() -> None:
    queries = sys.argv[1:] or SAMPLE_QUERIES
    for q in queries:
        await run_one(q)


if __name__ == "__main__":
    asyncio.run(main())
