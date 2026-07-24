"""text_to_sql 生成节点。"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.finance_agent.financial_query_agent.text_to_sql.generation.prompts import (
    FINANCIAL_QUERY_TEXT_TO_SQL_PROMPT,
)
from app.core.logger import get_logger
from agents.finance_agent.financial_query_agent.services.schemas import GeneratedFinancialSql
from agents.finance_agent.financial_query_agent.services.errors import classify_exception

logger = get_logger(service="financial_query")


def build_text_to_sql_prompt(*, schema_prompt: str, fewshot_examples: str) -> str:
    return FINANCIAL_QUERY_TEXT_TO_SQL_PROMPT.format(
        schema_prompt=schema_prompt,
        fewshot_examples=fewshot_examples,
    )


async def generate_sql(
    question: str,
    *,
    schema_prompt: str,
    fewshot_examples: str,
    config: RunnableConfig = None,
) -> GeneratedFinancialSql:
    """基于用户问题、Schema 和 few-shot 直接生成 SQL。"""
    fallback = GeneratedFinancialSql(
        sql="",
        params={},
        reason="复杂问题暂未生成可执行 SQL。",
        route="clarify",
        missing_fields=[],
    )
    try:
        llm = get_router_llm()
        system_prompt = build_text_to_sql_prompt(
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
                    ("human", f"用户问题：{question}"),
                ],
                config=config,
            ),
        )
    except Exception as exc:
        logger.exception("text_to_sql_agent sql generation failed")
        failure = classify_exception(exc, source="llm_generation")
        return fallback.model_copy(
            update={
                "failure_category": failure.category,
                "failure_code": failure.code,
                "failure_retryable": failure.retryable,
            }
        )


__all__ = ["build_text_to_sql_prompt", "generate_sql"]
