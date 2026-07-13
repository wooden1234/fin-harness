"""LangGraph Checkpoint：PostgresSaver + thread_id = conversation_id（Week 3 Day 5）。"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.core.config import settings
from app.core.logger import get_logger

# 允许 msgpack 反序列化项目内 Pydantic 状态模型（消除 checkpoint warning）。
# SAFE_MSGPACK_TYPES（messages / datetime 等）仍由 LangGraph 内置放行。
_CHECKPOINT_ALLOWED_MSGPACK_MODULES = (
    ("app.shared", "SubTask"),
    ("app.shared", "PlannerOutput"),
    ("agents.finance_agent.financial_query_agent.predefined.intent", "FinancialQueryIntent"),
)

# 兼容旧环境变量；未显式登记的类型在 strict 模式下会被拒绝。
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

logger = get_logger(service="checkpoint")

CheckpointBackend = Literal["postgres", "memory"]

_checkpointer: BaseCheckpointSaver | None = None
_exit_stack: AsyncExitStack | None = None


def normalize_checkpoint_dsn(url: str) -> str:
    """转为 psycopg 可识别的 ``postgresql://`` DSN。"""
    normalized = url.strip()
    if normalized.startswith("postgresql+asyncpg://"):
        normalized = "postgresql://" + normalized.removeprefix("postgresql+asyncpg://")
    return normalized


def checkpoint_dsn() -> str:
    # checkpoint 允许使用独立 DSN，这样可单独指定 runtime schema 的 search_path。
    raw = settings.LANGGRAPH_CHECKPOINT_URL or settings.PGVECTOR_DATABASE_URL
    if not raw:
        raise RuntimeError(
            "未配置 LANGGRAPH_CHECKPOINT_URL 或 PGVECTOR_DATABASE_URL，无法启用 PostgresSaver；"
            "推荐单独配置 LANGGRAPH_CHECKPOINT_URL 并带 search_path=runtime"
        )
    return normalize_checkpoint_dsn(raw)


def make_thread_config(conversation_id: str | int) -> RunnableConfig:
    """LangGraph 多轮：``thread_id`` 与业务 ``conversation_id`` 一一对应。"""
    return {"configurable": {"thread_id": str(conversation_id)}}


def _make_checkpoint_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=_CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    )


async def init_checkpoint(
    backend: CheckpointBackend | None = None,
) -> BaseCheckpointSaver:
    """初始化 checkpointer 并建表（Postgres 首次 ``setup()``）。"""
    global _checkpointer, _exit_stack

    if _checkpointer is not None:
        return _checkpointer

    chosen: CheckpointBackend = backend or settings.AGENT_CHECKPOINT_BACKEND  # type: ignore[assignment]

    if chosen == "memory":
        _checkpointer = MemorySaver(serde=_make_checkpoint_serde())
        logger.info("Agent checkpoint backend=memory")
        return _checkpointer

    _exit_stack = AsyncExitStack()
    saver = await _exit_stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(
            checkpoint_dsn(),
            serde=_make_checkpoint_serde(),
        )
    )
    await saver.setup()
    _checkpointer = saver
    logger.info("Agent checkpoint backend=postgres tables ready")
    return _checkpointer


async def close_checkpoint() -> None:
    """释放连接池（应用关闭时调用）。"""
    global _checkpointer, _exit_stack

    if _exit_stack is not None:
        await _exit_stack.aclose()
    _exit_stack = None
    _checkpointer = None

    from agents.graph import reset_graph_cache

    reset_graph_cache()
    logger.info("Agent checkpoint closed")


def get_checkpointer() -> BaseCheckpointSaver:
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer 未初始化，请在应用 lifespan 或脚本中先 await init_checkpoint()"
        )
    return _checkpointer
