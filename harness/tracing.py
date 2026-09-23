"""LangSmith 追踪：父 span 靠运行时上下文，thread_id 只做对话分组。

进程内：用 ``@traceable`` 包编排函数，SDK 自己生成根 run；
模型 / 工具从上下文读到父级，自己填 parent_run_id，不要手写 id。

跨进程：调用方 ``current_trace_headers()``，被调方
``incoming_trace_context(headers, thread_id=...)`` 接回父树。

``thread_id`` 不是父 id，不改变层级。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, tracing_context

_TRACE_HEADER_KEYS = frozenset({"langsmith-trace", "baggage"})


def thread_metadata(thread_id: str) -> dict[str, str]:
    """同一段对话的分组标签，不是树节点。"""
    value = str(thread_id or "").strip()
    return {"thread_id": value, "session_id": value} if value else {}


def turn_langsmith_extra(thread_id: str, *, entry: str) -> dict[str, Any]:
    """传给 @traceable 的 langsmith_extra。不包含 parent / run_id。"""
    return {
        "name": "agent.turn",
        "metadata": {**thread_metadata(thread_id), "entry": entry},
    }


def distributed_parent(headers: Mapping[str, str] | None) -> dict[str, str] | None:
    """从请求头取出 run tree 身份；没有就不装父级。"""
    if not headers:
        return None
    picked: dict[str, str] = {}
    for key, value in headers.items():
        if str(key).lower() in _TRACE_HEADER_KEYS and value:
            picked[str(key)] = str(value)
    return picked or None


def current_trace_headers() -> dict[str, str]:
    """把当前 run tree 放进 header，给另一个进程接。"""
    tree = get_current_run_tree()
    if tree is None:
        return {}
    return dict(tree.to_headers())


@contextmanager
def incoming_trace_context(
    headers: Mapping[str, str] | None,
    *,
    thread_id: str,
) -> Iterator[None]:
    """跨请求接回父树，并带上 thread_id。没有 header 时只设分组标签。"""
    kwargs: dict[str, Any] = {"metadata": thread_metadata(thread_id)}
    parent = distributed_parent(headers)
    if parent is not None:
        kwargs["parent"] = parent
    with tracing_context(**kwargs):
        yield


def without_self(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "self"}


__all__ = [
    "current_trace_headers",
    "distributed_parent",
    "incoming_trace_context",
    "thread_metadata",
    "traceable",
    "turn_langsmith_extra",
    "without_self",
]
