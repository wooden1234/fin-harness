"""Agent 执行进度：仅对外暴露面向用户的关键步骤。"""

from __future__ import annotations

# 用户可见的三类关键步骤（其余内部节点一律不展示）
PUBLIC_STEPS: dict[str, dict[str, str]] = {
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
}

# 内部节点 → 用户可见步骤（未映射的节点不推送 SSE）
NODE_TO_PUBLIC_STEP: dict[str, str] = {
    "financial_query_agent": "data_query",
    "faq_agent": "knowledge_base",
    "pdf_agent": "knowledge_base",
    "web_search_agent": "web_search",
    "summarize": "generating_answer",
    "general_agent": "generating_answer",
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
