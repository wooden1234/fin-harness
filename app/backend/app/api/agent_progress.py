"""Agent 执行进度：仅对外暴露面向用户的关键步骤。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from agents.main_deep_agent.state import normalize_agent_todos

# 用户可见的关键步骤（内部模型推理过程一律不展示）
PUBLIC_STEPS: dict[str, dict[str, str]] = {
    "problem_analysis": {
        "label_running": "正在分析问题并选择资料",
        "label_done": "已完成问题分析",
        "short": "问题分析",
    },
    "data_query": {
        "label_running": "正在查询数据表",
        "label_done": "已查询数据表",
        "short": "金融数据",
    },
    "knowledge_base": {
        "label_running": "正在查找相关知识库",
        "label_done": "已查找相关知识库",
        "short": "知识库",
    },
    "web_search": {
        "label_running": "正在通过搜索进行查找",
        "label_done": "已通过搜索进行查找",
        "short": "联网搜索",
    },
    "generating_answer": {
        "label_running": "正在生成答案",
        "label_done": "已生成答案",
        "short": "生成答案",
    },
    "evidence_validation": {
        "label_running": "正在验证证据与结论",
        "label_done": "已完成证据验证",
        "short": "证据验证",
    },
}

# 内部节点 → 用户可见步骤（未映射的节点不推送 SSE）
NODE_TO_PUBLIC_STEP: dict[str, str] = {
    "main_deep_agent": "problem_analysis",
    "evidence_quality_gate": "evidence_validation",
    "financial_query_agent": "data_query",
    "faq_agent": "knowledge_base",
    "pdf_agent": "knowledge_base",
    "web_search_agent": "web_search",
    "summarize": "generating_answer",
    "general_agent": "generating_answer",
    "final_answer": "generating_answer",
}

VISIBLE_TASK_NODES = frozenset(NODE_TO_PUBLIC_STEP.keys())


def map_node_to_public_step(node_name: str) -> str | None:
    return NODE_TO_PUBLIC_STEP.get(node_name)


def label_for_public_step(step_key: str, status: str) -> str:
    meta = PUBLIC_STEPS.get(step_key, {})
    if status == "running":
        return meta.get("label_running", step_key)
    return meta.get("label_done", step_key)


def short_label_for_public_step(step_key: str) -> str:
    return PUBLIC_STEPS.get(step_key, {}).get("short", step_key)


def build_step_event(
    *,
    step_id: str,
    label: str,
    status: str,
    category: str | None = None,
    short_label: str | None = None,
) -> dict:
    payload: dict = {
        "type": "step",
        "id": step_id,
        "label": label,
        "status": status,
    }
    if category:
        payload["category"] = category
    if short_label:
        payload["short_label"] = short_label
    return payload


def build_public_step_event(step_key: str, status: str) -> dict | None:
    if step_key not in PUBLIC_STEPS:
        return None
    return build_step_event(
        step_id=step_key,
        label=label_for_public_step(step_key, status),
        status=status,
        category=step_key,
        short_label=short_label_for_public_step(step_key),
    )


def extract_agent_todos_snapshot(
    value: object,
    *,
    depth: int = 0,
) -> list[dict[str, str]] | None:
    """从 Main DeepAgent 子图 update 或最终 Journal 中提取 todo 全量快照。"""
    if depth > 6 or not isinstance(value, Mapping):
        return None
    for key in ("todos", "agent_todos"):
        if key in value:
            return normalize_agent_todos(value.get(key))
    for nested in value.values():
        snapshot = extract_agent_todos_snapshot(nested, depth=depth + 1)
        if snapshot is not None:
            return snapshot
    return None


def build_todo_snapshot_event(raw_todos: object) -> dict:
    """构造全量替换事件，稳定 ID 不暴露 todo 原文。"""
    todos = normalize_agent_todos(raw_todos)
    occurrences: dict[str, int] = {}
    payload: list[dict[str, str]] = []
    for todo in todos:
        content = todo["content"]
        occurrences[content] = occurrences.get(content, 0) + 1
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        payload.append(
            {
                "id": f"main-todo-{digest}-{occurrences[content]}",
                "content": content,
                "status": todo["status"],
            }
        )
    return {"type": "todo_snapshot", "todos": payload}
