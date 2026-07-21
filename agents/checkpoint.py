"""LangGraph Checkpoint：按用户/租户隔离 thread_id。"""

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


def make_thread_id(
    conversation_id: str | int,
    *,
    user_id: str | int | None = None,
    tenant_id: str | int | None = None,
) -> str:
    """生成与创建时完全一致的 tenant-aware thread_id。

    格式：
    - ``tenant:{tenant_id}:conversation:{conversation_id}``
    - ``user:{user_id}:conversation:{conversation_id}``
    - ``anonymous:conversation:{conversation_id}``（仅本地/无身份）
    """
    if tenant_id is not None:
        scope = f"tenant:{tenant_id}"
    elif user_id is not None:
        scope = f"user:{user_id}"
    else:
        scope = "anonymous"
    return f"{scope}:conversation:{conversation_id}"


def make_thread_config(
    conversation_id: str | int,
    *,
    user_id: str | int | None = None,
    tenant_id: str | int | None = None,
) -> RunnableConfig:
    """生成隔离的 LangGraph thread config。

    API 请求必须传入 user_id；未传 user_id 的调用仅用于本地脚本或无身份
    的内部运行，并使用独立 anonymous 命名空间，禁止再直接使用业务主键。
    """
    thread_id = make_thread_id(
        conversation_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )
    return {"configurable": {"thread_id": thread_id}}


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


async def delete_thread_checkpoint(
    conversation_id: str | int,
    *,
    user_id: str | int | None = None,
    tenant_id: str | int | None = None,
) -> str:
    """按与创建时一致的 thread_id 删除该会话的全部 checkpoint / writes。

    Returns:
        实际删除使用的 ``thread_id``（便于审计日志）。
    """
    thread_id = make_thread_id(
        conversation_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )
    checkpointer = get_checkpointer()
    await checkpointer.adelete_thread(thread_id)
    logger.info("deleted checkpoint thread_id={}", thread_id)
    return thread_id
