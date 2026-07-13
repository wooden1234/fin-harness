"""Summarize 节点：跨源证据融合（仅负责汇总，收齐由 join 保证）。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.llm import get_faq_llm
from agents.states import FinAgentState, TaskResult
from agents.finance_agent.summarize.prompts import (
    CLARIFY_SUMMARY_PREFIX,
    MULTI_TASK_HUMAN_PROMPT,
    SINGLE_TASK_HUMAN_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    UNCOVERED_SUMMARY,
)
from app.core.logger import get_logger

logger = get_logger(service="summarize")

_COVERAGE_RANK = {"covered": 4, "partial": 3, "clarify": 2, "uncovered": 1}


def _coverage_of(result: TaskResult) -> str:
    if result.get("fallback_to_web"):
        return "uncovered"
    return str(result.get("coverage") or "covered")


def _pick_best_per_subtask(task_results: list[TaskResult]) -> list[TaskResult]:
    """同一 sub_task_id 多条结果时，保留覆盖度最高的一条（链走完后的 web 优于 faq uncovered）。"""
    best: dict[str, TaskResult] = {}
    for result in task_results:
        sub_id = str(result.get("sub_task_id") or "")
        key = sub_id or f"__anon_{result.get('type')}_{result.get('question')}"
        current = best.get(key)
        if current is None:
            best[key] = result
            continue
        if _COVERAGE_RANK.get(_coverage_of(result), 0) >= _COVERAGE_RANK.get(
            _coverage_of(current), 0
        ):
            best[key] = result
    return list(best.values())


def _format_task_results(task_results: list[TaskResult]) -> str:
    parts = []
    for i, tr in enumerate(task_results, start=1):
        coverage = _coverage_of(tr)
        parts.append(f"### 子任务 {i}：{tr.get('question', '未知')}")
        parts.append(f"类型：{tr.get('type', 'faq')}")
        parts.append(f"覆盖状态：{coverage}")
        if coverage == "uncovered":
            parts.append("说明：证据不足，请勿编造，应明确告知用户无可靠依据")
        elif coverage == "clarify":
            parts.append("说明：需向用户澄清，请转述澄清问题")
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

    current_ids = {t.id for t in sub_tasks if t.id}
    if current_ids:
        before = len(task_results)
        task_results = [tr for tr in task_results if tr.get("sub_task_id") in current_ids]
        if len(task_results) < before:
            logger.info("summarize filtered {} stale task_result(s)", before - len(task_results))

    task_results = _pick_best_per_subtask(task_results)

    logger.info(
        "summarize task_results=%d coverages=%s risk_level=%s",
        len(task_results),
        [_coverage_of(tr) for tr in task_results],
        risk_level,
    )

    if risk_level in ("L3", "L4"):
        return {"summary": "您的问题涉及敏感内容，建议联系人工客服获得进一步帮助。"}

    if not task_results:
        return {"summary": ""}

    coverages = [_coverage_of(tr) for tr in task_results]
    if all(c == "uncovered" for c in coverages):
        return {"summary": UNCOVERED_SUMMARY}

    if len(task_results) == 1 and coverages[0] == "clarify":
        context = str(task_results[0].get("context") or "").strip()
        if context:
            return {"summary": f"{CLARIFY_SUMMARY_PREFIX}{context}"}
        return {"summary": UNCOVERED_SUMMARY}

    try:
        if len(task_results) == 1:
            tr = task_results[0]
            context = str(tr.get("context") or "").strip()
            if not context:
                return {"summary": UNCOVERED_SUMMARY if _coverage_of(tr) == "uncovered" else ""}
            if _coverage_of(tr) == "uncovered":
                return {"summary": UNCOVERED_SUMMARY}
            if tr.get("type") == "financial_query":
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
