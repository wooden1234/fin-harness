"""从业务消息重建 LangGraph checkpoint。"""

from __future__ import annotations

from sqlalchemy import select
from langchain_core.messages import AIMessage, HumanMessage

from agents.graph import get_graph
from agents.checkpoint import delete_thread_checkpoint
from app.core.database import AsyncSessionLocal
from app.models.message import Message


class CheckpointRebuildService:
    @staticmethod
    async def rebuild_if_missing(
        *,
        conversation_id: int,
        user_id: int,
        thread_config: dict,
        exclude_run_id: str | None = None,
    ) -> bool:
        """checkpoint 缺失时用历史消息恢复；返回是否实际重建。"""
        graph = get_graph()
        current = await graph.aget_state(thread_config)
        if current is not None and (current.values or {}).get("messages"):
            return False

        async with AsyncSessionLocal() as db:
            stmt = select(Message).where(
                Message.conversation_id == conversation_id
            )
            if exclude_run_id:
                stmt = stmt.where(Message.run_id != exclude_run_id)
            stmt = stmt.order_by(Message.sequence_no, Message.id)
            rows = (await db.execute(stmt)).scalars().all()

        messages = []
        for row in rows:
            if row.sender == "user":
                messages.append(HumanMessage(content=row.content))
            elif row.sender == "assistant":
                messages.append(AIMessage(content=row.content))
        if not messages:
            return False

        await delete_thread_checkpoint(conversation_id, user_id=user_id)
        await graph.aupdate_state(thread_config, {"messages": messages})
        return True

