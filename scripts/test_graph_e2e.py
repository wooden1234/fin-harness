"""主图端到端联调：Supervisor → FAQ。

用法:
    source .venv/bin/activate
    python scripts/test_graph_e2e.py
    python scripts/test_graph_e2e.py "什么是 T+1 交易制度？"
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env")

from agents.checkpoint import close_checkpoint, init_checkpoint, make_thread_config  # noqa: E402
from agents.graph import get_graph  # noqa: E402

DEFAULT_QUERY = "什么是 T+1 交易制度？"


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    await init_checkpoint(backend="postgres")
    graph = get_graph()
    conversation_id = str(uuid.uuid4())
    config = make_thread_config(conversation_id)

    print(f"conversation_id (thread_id)={conversation_id}")
    print(f"Q: {query}\n")

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    messages = result.get("messages") or []
    last = messages[-1] if messages else None
    answer = getattr(last, "content", "") if last else ""

    print("--- route / risk ---")
    print(f"route      = {result.get('route')}")
    print(f"risk_level = {result.get('risk_level')}")
    print(f"logic      = {result.get('logic')}")
    print("\n--- answer ---")
    print(answer)
    print("\n--- citations ---")
    for i, c in enumerate(result.get("citations") or [], start=1):
        print(f"  [{i}] {c.get('source')} | {(c.get('snippet') or '')[:80]}...")

    await close_checkpoint()


if __name__ == "__main__":
    asyncio.run(main())
