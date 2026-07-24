"""text_to_sql 状态机业务节点。"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.services.errors import (
    classify_exception,
    classify_sql_validation,
    classify_unsupported,
    classify_user_clarification,
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
from agents.finance_agent.financial_query_agent.text_to_sql.components.retry_guard import (
    resolve_retry_action,
)
from agents.finance_agent.financial_query_agent.text_to_sql.validation import (
    validate_generated_sql,
    validate_query_result_full,
)
from app.core.logger import get_logger

logger = get_logger(service="financial_query")

DEFAULT_TEXT_TO_SQL_TOP_K = 5
# 总尝试次数为 2：首次生成 1 次，最多纠错 1 次。
DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS = 2
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
    current_state, halt_result = await _MIDDLEWARE_CHAIN.run_before_generate(
        {"question": question},
        config,
    )
    if halt_result:
        next_step = (
            "execution_error_output"
            if halt_result.halt_reason == "unsupported"
            else "clarify_output"
        )
        failure = (
            classify_unsupported(halt_result.halt_reason)
            if halt_result.halt_reason == "unsupported"
            else classify_user_clarification()
        )
        return {
            **current_state,
            **halt_updates(halt_result),
            "failure_category": failure.category,
            "failure_code": failure.code,
            "failure_retryable": failure.retryable,
            "next_step": next_step,
        }

    requested_attempts = int(
        state.get("max_attempts", DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS)
    )
    context = {
        "question": question,
        "schema_prompt": build_schema_prompt(question),
        "fewshot_examples": build_fewshot_examples(question),
        "top_k": int(state.get("top_k", DEFAULT_TEXT_TO_SQL_TOP_K)),
        "max_attempts": min(
            max(requested_attempts, 1),
            DEFAULT_TEXT_TO_SQL_MAX_ATTEMPTS,
        ),
        "attempts": int(state.get("attempts", 0)),
        "seen_sql_hashes": list(state.get("seen_sql_hashes", [])),
        "last_error_type": str(state.get("last_error_type", "")),
        "repeat_error_count": int(state.get("repeat_error_count", 0)),
        "next_step": "generate_sql",
    }
    context.update(current_state)
    return context


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
        "query_contract": generated.query_contract,
        "failure_category": generated.failure_category,
        "failure_code": generated.failure_code,
        "failure_retryable": generated.failure_retryable,
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
        query_contract=state.get("query_contract"),
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
    """规则门：拦截简单危险错误；通过后进入真库验证。"""
    attempts = int(state.get("attempts", 0)) + 1
    validation = validate_generated_sql(
        state.get("sql", ""),
        params=state.get("sql_params", {}),
    )
    validation_errors = [validation.error] if validation.error else []
    validation_error_types = [validation.error_type] if validation.error_type else []
    if not validation.ok:
        retry_state = {**state, "attempts": attempts}
        return {
            "attempts": attempts,
            "validated_sql": validation.validated_sql,
            "failure_category": classify_sql_validation(validation.error_type).category,
            "failure_code": classify_sql_validation(validation.error_type).code,
            "failure_retryable": False,
            **resolve_retry_action(
                retry_state,
                error_type=validation.error_type,
                error=validation.error,
                sql=state.get("sql", ""),
                params=state.get("sql_params", {}),
                terminal_on_abort="unsafe_output",
            ),
        }
    return {
        "attempts": attempts,
        "validated_sql": validation.validated_sql,
        "validation_error": validation.error,
        "validation_error_type": validation.error_type,
        "validation_errors": validation_errors,
        "validation_error_types": validation_error_types,
        "next_step": "db_verify",
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
        "query_contract": corrected.query_contract or state.get("query_contract"),
        "failure_category": corrected.failure_category or state.get("failure_category"),
        "failure_code": corrected.failure_code or state.get("failure_code", ""),
        "failure_retryable": corrected.failure_retryable,
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
        query_contract=state.get("query_contract"),
    )
    halt_result = await _MIDDLEWARE_CHAIN.run_after_correct(state, corrected, config)
    if halt_result:
        return {
            **halt_updates(halt_result),
            "next_step": "clarify_output",
        }
    return {"next_step": "validate_sql"}


async def db_verify_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """真库验证：规则过不了的运行时/语义问题在此暴露；成功则进入结果校验。"""
    try:
        rows = await execute_generated_sql(
            state.get("validated_sql", "") or state.get("sql", ""),
            params=state.get("sql_params", {}),
            limit=int(state.get("top_k", DEFAULT_TEXT_TO_SQL_TOP_K)),
        )
        return {
            "rows": rows,
            "execution_error": "",
            "next_step": "validate_result",
        }
    except Exception as exc:
        logger.exception("text_to_sql_workflow db_verify failed")
        failure = classify_exception(exc, source="database_verify")
        error_text = failure.code
        failure_updates = {
            "failure_category": failure.category,
            "failure_code": failure.code,
            "failure_retryable": failure.retryable,
        }
        if failure.category in {
            "database_timeout",
            "database_unavailable",
            "schema_mismatch",
        }:
            return {
                "rows": [],
                "execution_error": error_text,
                "validation_error": error_text,
                "validation_error_type": "runtime",
                "validation_errors": [error_text],
                "validation_error_types": ["runtime"],
                **failure_updates,
                "next_step": "execution_error_output",
            }
        return {
            "rows": [],
            "execution_error": error_text,
            **failure_updates,
            **resolve_retry_action(
                state,
                error_type="runtime",
                error=error_text,
                sql=state.get("validated_sql", "") or state.get("sql", ""),
                params=state.get("sql_params", {}),
                terminal_on_abort="execution_error_output",
            ),
        }


async def validate_result_node(
    state: TextToSqlState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """结果校验：规则 → 可选 LLM 质检；失败走 correct_sql，重试用尽走 execution_error。"""
    rows = list(state.get("rows", []))
    validation = await validate_query_result_full(
        question=state.get("question", ""),
        sql=state.get("validated_sql", "") or state.get("sql", ""),
        rows=rows,
        contract=state.get("query_contract"),
        config=config,
    )
    if validation.ok:
        return {
            "result_validation_ok": True,
            "result_validation_error": "",
            "failure_category": validation.failure_category,
            "failure_code": validation.failure_code,
            "failure_retryable": validation.failure_retryable,
            "next_step": "format_output",
        }

    if validation.should_clarify:
        clarification = validation.error.removeprefix("结果质检失败：").strip()
        return {
            "result_validation_ok": False,
            "result_validation_error": validation.error,
            "halt_reason": "clarify",
            "halt_answer": clarification,
            "failure_category": classify_user_clarification().category,
            "failure_code": classify_user_clarification().code,
            "failure_retryable": False,
            "next_step": "clarify_output",
        }

    return {
        "result_validation_ok": False,
        "result_validation_error": validation.error,
        **resolve_retry_action(
            state,
            error_type=validation.error_type,
            error=validation.error,
            sql=state.get("validated_sql", "") or state.get("sql", ""),
            params=state.get("sql_params", {}),
            terminal_on_abort="execution_error_output",
        ),
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
        "failure_category": "sql_unsafe",
        "failure_code": "sql_unsafe",
        "failure_retryable": False,
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
        "failure_category": None,
        "failure_code": "",
        "failure_retryable": False,
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
    "clarify_output_node",
    "correct_sql_node",
    "db_verify_node",
    "execution_error_output_node",
    "format_output_node",
    "generate_sql_node",
    "prepare_context_node",
    "unsafe_output_node",
    "validate_result_node",
    "validate_sql_node",
]
