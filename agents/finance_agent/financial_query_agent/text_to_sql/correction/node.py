"""text_to_sql 修正节点。"""

from __future__ import annotations

import json
from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.finance_agent.financial_query_agent.text_to_sql.correction.prompts import (
    FINANCIAL_QUERY_TEXT_TO_SQL_CORRECTION_PROMPT,
)
from app.core.logger import get_logger
from agents.finance_agent.financial_query_agent.services.schemas import GeneratedFinancialSql

logger = get_logger(service="financial_query")


async def correct_sql(
    question: str,
    *,
    schema_prompt: str,
    fewshot_examples: str,
    sql: str,
    params: dict[str, object],
    validation_errors: list[str],
    validation_error_type: str = "",
    config: RunnableConfig = None,
) -> GeneratedFinancialSql:
    """根据校验错误修正 SQL，保留原始用户问题不变。"""
    fallback = GeneratedFinancialSql(
        sql=sql,
        params=params,
        reason="SQL 修正失败，保留原始结果。",
        route="clarify" if not sql.strip() else "execute",
        missing_fields=[],
    )
    try:
        llm = get_router_llm()
        system_prompt = FINANCIAL_QUERY_TEXT_TO_SQL_CORRECTION_PROMPT.format(
            schema_prompt=schema_prompt,
            fewshot_examples=fewshot_examples,
        )
        return cast(
            GeneratedFinancialSql,
            await llm.with_structured_output(
                GeneratedFinancialSql,
                method="json_mode",
            ).ainvoke(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        "用户问题：{question}\n原始 SQL：{sql}\n原始参数：{params}\n错误类型：{error_type}\n校验错误：{errors}".format(
                            question=question,
                            sql=sql,
                            params=json.dumps(params, ensure_ascii=False),
                            error_type=validation_error_type or "unknown",
                            errors=json.dumps(validation_errors, ensure_ascii=False),
                        ),
                    ),
                ],
                config=config,
            ),
        )
    except Exception:
        logger.exception("text_to_sql_agent sql correction failed")
        return fallback


__all__ = ["correct_sql"]
