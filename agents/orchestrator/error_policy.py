"""V2 错误分类与处理动作。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorAction = Literal["retry", "fallback", "clarify", "fail"]


@dataclass(frozen=True, slots=True)
class ErrorDecision:
    """一次任务错误的标准处理决策。"""

    action: ErrorAction
    error_code: str
    retryable: bool = False


_CLARIFY_MARKERS = (
    "missing",
    "ambiguous",
    "clarify",
    "not_allowed",
    "invalid_query",
)
_RETRY_MARKERS = (
    "timeout",
    "connection",
    "rate_limit",
    "ratelimit",
    "temporarily_unavailable",
    "service_unavailable",
    "provider_5xx",
)
_FALLBACK_CODES = frozenset(
    {
        "source_tool_failed",
        "research_sources_partial",
        "research_skill_missing",
        "faq_no_context",
        "pdf_graph_error",
        "web_search_failed",
        "llm_unavailable",
        "planner_llm_failed",
    }
)
_FAIL_MARKERS = (
    "permission",
    "forbidden",
    "schema",
    "invalid_tool",
    "tool_mismatch",
    "computation_failed",
    "not_implemented",
)


def classify_error(
    error_code: str | None = None,
    exc: BaseException | None = None,
) -> ErrorDecision:
    """把异常或 AgentResult 错误码收敛为 V2 处理动作。"""
    code = str(error_code or "").strip().lower()
    if not code and exc is not None:
        code = type(exc).__name__.strip().lower()
    if not code:
        code = "unknown_error"

    if code in {"run_soft_deadline_exceeded", "run_hard_deadline_exceeded"}:
        return ErrorDecision(
            action="fallback" if code.startswith("run_soft") else "fail",
            error_code=code,
        )
    if any(marker in code for marker in _CLARIFY_MARKERS):
        return ErrorDecision(action="clarify", error_code=code)
    if any(marker in code for marker in _FAIL_MARKERS):
        return ErrorDecision(action="fail", error_code=code)
    if code in _FALLBACK_CODES or any(
        marker in code for marker in ("unavailable", "empty_result")
    ):
        return ErrorDecision(action="fallback", error_code=code)
    if any(marker in code for marker in _RETRY_MARKERS):
        return ErrorDecision(action="retry", error_code=code, retryable=True)
    if exc is not None and isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return ErrorDecision(action="retry", error_code=code, retryable=True)
    return ErrorDecision(action="fail", error_code=code)


__all__ = ["ErrorAction", "ErrorDecision", "classify_error"]
