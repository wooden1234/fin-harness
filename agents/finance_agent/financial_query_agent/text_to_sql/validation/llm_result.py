"""text_to_sql 结果层 LLM 质检（可选）。"""

from __future__ import annotations

import json
from typing import cast, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.llm import get_router_llm
from agents.finance_agent.financial_query_agent.services.schemas import FinancialSqlResultRow
from agents.finance_agent.financial_query_agent.services.errors import classify_exception
from agents.finance_agent.financial_query_agent.text_to_sql.validation.result import ResultValidation
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

FINANCIAL_QUERY_TEXT_TO_SQL_LLM_RESULT_VALIDATION_PROMPT = """你是 financial_query 的 SQL 结果质检员。请判断查询结果是否在回答用户问题。

输入包含：用户问题、已执行 SQL、前几行查询结果。

判定标准：
1. ok：结果在语义上回答了用户问题（公司、指标、年份/范围、对比对象等大体正确）
2. wrong_sql：SQL 能跑且有数据，但明显答非所问（公司错、指标错、年份范围不对、对比缺对象等）——应修正 SQL
3. need_clarify：用户问题本身仍模糊，无法判断该选哪家公司/哪个口径/哪个年份——应追问用户

要求：
1. 只输出 JSON，不要 markdown
2. 用户题面已明确时，优先判 wrong_sql，不要滥用 need_clarify
3. reason 用一句简洁中文说明
"""

_LLM_RESULT_PREVIEW_ROWS = 5
_LLM_RESULT_SQL_MAX_CHARS = 2000


class LlmResultValidationDecision(BaseModel):
    verdict: Literal["ok", "wrong_sql", "need_clarify"] = Field(
        description="结果是否像在回答问题。",
    )
    reason: str = Field(default="", description="判定理由。")


def is_llm_result_validation_enabled() -> bool:
    return bool(settings.FINANCIAL_SQL_LLM_VALIDATION)


def _preview_rows(rows: list[FinancialSqlResultRow], *, limit: int = _LLM_RESULT_PREVIEW_ROWS) -> str:
    payload = [row.model_dump(exclude_none=True) for row in rows[:limit]]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _preview_sql(sql: str) -> str:
    normalized = sql.strip()
    if len(normalized) <= _LLM_RESULT_SQL_MAX_CHARS:
        return normalized
    return normalized[:_LLM_RESULT_SQL_MAX_CHARS] + "\n-- [truncated]"


async def validate_query_result_with_llm(
    *,
    question: str,
    sql: str,
    rows: list[FinancialSqlResultRow],
    config: RunnableConfig | None = None,
) -> ResultValidation:
    """规则通过后，用 LLM 判断结果是否像在答题。失败时 fail-open 放行。"""
    if not is_llm_result_validation_enabled():
        return ResultValidation.passed()

    try:
        llm = get_router_llm()
        decision = cast(
            LlmResultValidationDecision,
            await llm.with_structured_output(
                LlmResultValidationDecision,
                method="json_mode",
            ).ainvoke(
                [
                    ("system", FINANCIAL_QUERY_TEXT_TO_SQL_LLM_RESULT_VALIDATION_PROMPT),
                    (
                        "human",
                        (
                            f"用户问题：{question.strip()}\n"
                            f"SQL：\n{_preview_sql(sql)}\n"
                            f"查询结果（前 {min(len(rows), _LLM_RESULT_PREVIEW_ROWS)} 行）：\n"
                            f"{_preview_rows(rows)}"
                        ),
                    ),
                ],
                config=config,
            ),
        )
    except Exception as exc:
        logger.exception("text_to_sql llm result validation failed")
        failure = classify_exception(exc, source="llm_result_validation")
        return ResultValidation.passed(
            failure_category=failure.category,
            failure_code=failure.code,
            failure_retryable=failure.retryable,
        )

    if decision.verdict == "ok":
        return ResultValidation.passed()

    reason = decision.reason.strip() or "结果质检未通过：查询结果与用户问题不够一致。"
    if decision.verdict == "need_clarify":
        return ResultValidation(
            ok=False,
            error=f"结果质检失败：{reason}",
            error_type="semantic",
            should_clarify=True,
        )

    return ResultValidation(
        ok=False,
        error=f"结果质检失败：{reason}",
        error_type="semantic",
    )


__all__ = [
    "LlmResultValidationDecision",
    "is_llm_result_validation_enabled",
    "validate_query_result_with_llm",
]
