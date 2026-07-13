"""text_to_sql 状态机业务节点。"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.correction import (
    correct_sql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.execution import (
    execute_generated_sql,
    format_sql_rows,
)
from agents.finance_agent.financial_query_agent.text_to_sql.generation import (
    build_fewshot_examples,
    build_schema_prompt,
    generate_sql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.middleware import (
    default_middleware_chain,
    halt_updates,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlState,
)
from agents.finance_agent.financial_query_agent.text_to_sql.validation import (
    validate_generated_sql,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

DEFAULT_TEXT_TO_SQL_TOP_K = 5
DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS = 3
FINANCIAL_QUERY_SQL_UNSAFE_ANSWER = (
    "当前问题需要生成 SQL，但生成结果未通过只读安全校验。请补充更具体的查询条件后重试。"
)
FINANCIAL_QUERY_TEXT_TO_SQL_FALLBACK_ANSWER = (
    "当前问题超出安全模板和低风险通用搜索范围，建议回退到更灵活的查询规划，例如 text-to-SQL，或先将问题拆得更具体。"
)
FINANCIAL_QUERY_CLARIFY_ANSWER = "请补充更明确的公司名称、财务指标或统计年份后，我再继续生成查询。"

_MIDDLEWARE_CHAIN = default_middleware_chain()


async def prepare_context_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    question = state["question"].strip()
    return {
        "question": question,
        "schema_prompt": build_schema_prompt(),
        "fewshot_examples": build_fewshot_examples(question),
        "top_k": int(state.get("top_k", DEFAULT_TEXT_TO_SQL_TOP_K)),
        "max_attempts": int(state.get("max_attempts", DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS)),
        "attempts": int(state.get("attempts", 0)),
        "next_step": "clarify_before_generate",
    }


async def clarify_before_generate_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    current_state, halt_result = await _MIDDLEWARE_CHAIN.run_before_generate(state, config)
    if halt_result:
        return {
            **halt_updates(halt_result),
            "next_step": "clarify_output",
        }
    return {**current_state, "next_step": "generate_sql"}


async def generate_sql_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    generated = await generate_sql(
        state["question"],
        schema_prompt=state["schema_prompt"],
        fewshot_examples=state["fewshot_examples"],
        config=config,
    )
    return {
        "sql": generated.sql,
        "sql_params": generated.params,
        "sql_route": generated.route,
        "route_reason": generated.reason,
        "missing_fields": generated.missing_fields,
        "next_step": "clarify_after_generate",
    }


async def clarify_after_generate_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    generated = GeneratedFinancialSql(
        sql=state.get("sql", ""),
        params=state.get("sql_params", {}),
        reason=state.get("route_reason", ""),
        route=state.get("sql_route", "execute"),
        missing_fields=state.get("missing_fields", []),
    )
    halt_result = await _MIDDLEWARE_CHAIN.run_after_generate(state, generated, config)
    if halt_result:
        return {
            **halt_updates(halt_result),
            "next_step": "clarify_output",
        }
    return {"next_step": "validate_sql"}


async def validate_sql_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    attempts = int(state.get("attempts", 0)) + 1
    validation = validate_generated_sql(
        state.get("sql", ""),
        params=state.get("sql_params", {}),
    )
    validation_errors = [validation.error] if validation.error else []
    validation_error_types = [validation.error_type] if validation.error_type else []
    next_step = "execute_sql"
    if not validation.ok:
        max_attempts = int(state.get("max_attempts", DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS))
        next_step = "unsafe_output" if attempts >= max_attempts else "correct_sql"
    return {
        "attempts": attempts,
        "validated_sql": validation.validated_sql,
        "validation_error": validation.error,
        "validation_error_type": validation.error_type,
        "validation_errors": validation_errors,
        "validation_error_types": validation_error_types,
        "next_step": next_step,
    }


async def correct_sql_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    corrected = await correct_sql(
        state["question"],
        schema_prompt=state["schema_prompt"],
        fewshot_examples=state["fewshot_examples"],
        sql=state.get("sql", ""),
        params=state.get("sql_params", {}),
        validation_errors=state.get("validation_errors", []),
        validation_error_type=state.get("validation_error_type", ""),
        config=config,
    )
    return {
        "sql": corrected.sql,
        "sql_params": corrected.params,
        "sql_route": corrected.route,
        "route_reason": corrected.reason,
        "missing_fields": corrected.missing_fields,
        "next_step": "clarify_after_correct",
    }


async def clarify_after_correct_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    corrected = GeneratedFinancialSql(
        sql=state.get("sql", ""),
        params=state.get("sql_params", {}),
        reason=state.get("route_reason", ""),
        route=state.get("sql_route", "execute"),
        missing_fields=state.get("missing_fields", []),
    )
    halt_result = await _MIDDLEWARE_CHAIN.run_after_correct(state, corrected, config)
    if halt_result:
        return {
            **halt_updates(halt_result),
            "next_step": "clarify_output",
        }
    return {"next_step": "validate_sql"}


async def execute_sql_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    try:
        rows = await execute_generated_sql(
            state.get("validated_sql", "") or state.get("sql", ""),
            params=state.get("sql_params", {}),
            limit=int(state.get("top_k", DEFAULT_TEXT_TO_SQL_TOP_K)),
        )
        return {
            "rows": rows,
            "execution_error": "",
            "next_step": "format_output",
        }
    except Exception:
        logger.exception("text_to_sql_workflow execution failed")
        return {
            "rows": [],
            "execution_error": "sql_execution_failed",
            "next_step": "execution_error_output",
        }


async def clarify_output_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    return {
        "answer": state.get("halt_answer", "") or FINANCIAL_QUERY_CLARIFY_ANSWER,
        "final_status": "clarify",
        "next_step": "end",
    }


async def unsafe_output_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    return {
        "answer": FINANCIAL_QUERY_SQL_UNSAFE_ANSWER,
        "final_status": "unsafe",
        "next_step": "end",
    }


async def execution_error_output_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    return {
        "answer": FINANCIAL_QUERY_TEXT_TO_SQL_FALLBACK_ANSWER,
        "final_status": "execution_error",
        "next_step": "end",
    }


async def format_output_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    return {
        "answer": format_sql_rows(list(state.get("rows", []))),
        "final_status": "success",
        "next_step": "end",
    }


__all__ = [
    "DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS",
    "DEFAULT_TEXT_TO_SQL_TOP_K",
    "FINANCIAL_QUERY_CLARIFY_ANSWER",
    "FINANCIAL_QUERY_SQL_UNSAFE_ANSWER",
    "FINANCIAL_QUERY_TEXT_TO_SQL_FALLBACK_ANSWER",
    "clarify_after_correct_node",
    "clarify_after_generate_node",
    "clarify_before_generate_node",
    "clarify_output_node",
    "correct_sql_node",
    "execute_sql_node",
    "execution_error_output_node",
    "format_output_node",
    "generate_sql_node",
    "prepare_context_node",
    "unsafe_output_node",
    "validate_sql_node",
]
