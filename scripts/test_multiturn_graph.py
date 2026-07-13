"""多轮对话联调：同一 conversation_id（thread_id）续聊带上文。

用法:
    source .venv/bin/activate
    python scripts/test_multiturn_graph.py
    python scripts/test_multiturn_graph.py 42   # 指定 conversation_id
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.agents.checkpoint import close_checkpoint, init_checkpoint, make_thread_config  # noqa: E402
from app.agents.graph import get_graph, reset_graph_cache  # noqa: E402

TURN1 = "什么是 T+1 交易制度？"
TURN2 = "刚才说的这个制度，当天买入的股票什么时候可以卖？"


async def main() -> None:
    conversation_id = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())
    await init_checkpoint(backend="postgres")
    graph = get_graph()
    config = make_thread_config(conversation_id)

    print(f"conversation_id (thread_id) = {conversation_id}\n")

    print(f"[Turn 1] Q: {TURN1}")
    r1 = await graph.ainvoke({"messages": [HumanMessage(content=TURN1)]}, config=config)
    _print_turn(r1)

    print(f"\n[Turn 2] Q: {TURN2}")
    r2 = await graph.ainvoke({"messages": [HumanMessage(content=TURN2)]}, config=config)
    _print_turn(r2)

    msg_count = len(r2.get("messages") or [])
    print(f"\n--- multi-turn check ---")
    print(f"total messages in state = {msg_count} (expect >= 4: 2 user + 2 assistant)")
    if msg_count < 4:
        print("WARN: 消息数偏少，checkpoint 可能未正确续聊")

    await close_checkpoint()


def _print_turn(result: dict) -> None:
    messages = result.get("messages") or []
    last = messages[-1] if messages else None
    answer = last.content if isinstance(last, AIMessage) else getattr(last, "content", "")
    print(f"  route={result.get('route')} risk={result.get('risk_level')}")
    print(f"  messages={len(messages)}")
    print(f"  A: {str(answer)[:300]}...")


if __name__ == "__main__":
    asyncio.run(main())
