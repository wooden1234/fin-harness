"""Agent 执行进度：仅对外暴露面向用户的关键步骤。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from agents.main_deep_agent.state import normalize_agent_todos

# 用户可见的关键步骤（内部模型推理过程一律不展示）
PUBLIC_STEPS: dict[str, dict[str, str]] = {
    "problem_analysis": {
        "label_running": "正在理解问题…",
        "label_done": "已理解问题",
        "short": "问题理解",
    },
    "data_query": {
        "label_running": "正在查询财务数据…",
        "label_done": "已取得财务数据",
        "short": "财务数据",
    },
    "knowledge_base": {
        "label_running": "正在查阅知识库…",
        "label_done": "已查阅知识库",
        "short": "知识库",
    },
    "web_search": {
        "label_running": "正在检索联网资料…",
        "label_done": "已取得联网资料",
        "short": "联网资料",
    },
    "generating_answer": {
        "label_running": "正在整理答案…",
        "label_done": "已整理答案",
        "short": "整理答案",
    },
    "evidence_validation": {
        "label_running": "正在核对证据…",
        "label_done": "已核对证据",
        "short": "核对证据",
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


def sanitize_step_detail(raw: object) -> dict | None:
    """只保留前端可渲染的展示字段，拒绝透传原始工具 payload。"""
    if not isinstance(raw, Mapping):
        return None
    detail: dict = {}
    title = str(raw.get("title") or "").strip()[:80]
    query = str(raw.get("query") or "").strip()[:200]
    display_text = str(raw.get("display_text") or "").strip()[:800]
    error = str(raw.get("error") or "").strip()[:120]
    if title:
        detail["title"] = title
    if query:
        detail["query"] = query
    if display_text and not display_text.startswith(("{", "[")):
        detail["display_text"] = display_text
    if error:
        detail["error"] = error
    columns = raw.get("columns")
    rows = raw.get("rows")
    if isinstance(columns, list) and isinstance(rows, list) and columns and rows:
        clean_columns = [str(item).strip()[:40] for item in columns[:6] if str(item).strip()]
        clean_rows: list[list[str]] = []
        for row in rows[:8]:
            if not isinstance(row, (list, tuple)):
                continue
            cells = [str(cell).strip()[:80] for cell in list(row)[: len(clean_columns)]]
            if len(cells) == len(clean_columns) and any(cells):
                clean_rows.append(cells)
        if clean_columns and clean_rows:
            detail["columns"] = clean_columns
            detail["rows"] = clean_rows
    return detail or None


def build_step_event(
    *,
    step_id: str,
    label: str,
    status: str,
    category: str | None = None,
    short_label: str | None = None,
    detail: object | None = None,
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
    cleaned = sanitize_step_detail(detail)
    if cleaned:
        payload["detail"] = cleaned
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
