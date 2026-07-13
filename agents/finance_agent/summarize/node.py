"""Summarize 节点：跨源证据融合（仅负责汇总，收齐由 join 保证）。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.llm import get_faq_llm
from agents.states import FinAgentState, TaskResult
from agents.finance_agent.summarize.prompts import (
    MULTI_TASK_HUMAN_PROMPT,
    SINGLE_TASK_HUMAN_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
)
from app.core.logger import get_logger

logger = get_logger(service="summarize")


def _format_task_results(task_results: list[TaskResult]) -> str:
    parts = []
    for i, tr in enumerate(task_results, start=1):
        parts.append(f"### 子任务 {i}：{tr.get('question', '未知')}")
        parts.append(f"类型：{tr.get('type', 'faq')}")
        parts.append(f"结果：{tr.get('context', '无结果')}")
        parts.append("")
    return "\n".join(parts) if parts else "无检索结果"


async def _stream_llm_summary(
    risk_level: str,
    human_content: str,
    config: RunnableConfig | None,
) -> str:
    llm = get_faq_llm()
    parts: list[str] = []
    async for chunk in llm.astream(
        [
            ("system", SUMMARIZE_SYSTEM_PROMPT.format(risk_level=risk_level)),
            ("human", human_content),
        ],
        config=config,
    ):
        if chunk.content:
            parts.append(chunk.content if isinstance(chunk.content, str) else str(chunk.content))
    return "".join(parts)


async def summarize_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    task_results: list[TaskResult] = list(state.get("task_results") or [])
    sub_tasks = list(state.get("sub_tasks") or [])
    risk_level = state.get("risk_level", "L1")

    # 仅保留当前轮次子任务的结果（sub_tasks 不跨轮累积，用其 ID 过滤）
    current_ids = {t.id for t in sub_tasks if t.id}
    if current_ids:
        before = len(task_results)
        task_results = [tr for tr in task_results if tr.get("sub_task_id") in current_ids]
        if len(task_results) < before:
            logger.info("summarize filtered {} stale task_result(s)", before - len(task_results))

    if any(tr.get("type") == "web_search" for tr in task_results):
        task_results = [tr for tr in task_results if not tr.get("fallback_to_web")]

    logger.info("summarize task_results=%d risk_level=%s", len(task_results), risk_level)

    if risk_level in ("L3", "L4"):
        return {"summary": "您的问题涉及敏感内容，建议联系人工客服获得进一步帮助。"}

    if not task_results:
        return {"summary": ""}

    try:
        if len(task_results) == 1:
            context = task_results[0].get("context", "")
            if not context.strip():
                return {"summary": ""}
            if task_results[0].get("type") == "financial_query":
                human = SINGLE_TASK_HUMAN_PROMPT.format(context=context)
                summary = await _stream_llm_summary(risk_level, human, config)
            else:
                summary = context
        else:
            formatted = _format_task_results(task_results)
            human = MULTI_TASK_HUMAN_PROMPT.format(formatted=formatted)
            summary = await _stream_llm_summary(risk_level, human, config)
    except Exception:
        logger.exception("summarize llm invoke failed")
        summary = "抱歉，在汇总信息时出现错误，请稍后重试。"

    return {"summary": summary}
