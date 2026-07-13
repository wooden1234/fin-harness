"""text_to_sql 纠错环防无限循环：SQL 指纹去重 + 连续相同错误提前放弃。

当前用 TextToSqlState 内存列表；后期可替换为 Redis 实现 SqlRetryGuardStore。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol

from app.core.config import settings
from agents.finance_agent.financial_query_agent.text_to_sql.state import TextToSqlState
from agents.finance_agent.financial_query_agent.text_to_sql.validation.node import ValidationErrorType

RetryTerminal = Literal["unsafe_output", "execution_error_output"]


class SqlRetryGuardStore(Protocol):
    """后期 Redis 缓存可实现此协议，按 session/thread 维度存 seen_sql。"""

    async def has_sql(self, scope: str, fingerprint: str) -> bool: ...

    async def remember_sql(self, scope: str, fingerprint: str) -> None: ...


class StateSqlRetryGuardStore:
    """进程内实现：读写 TextToSqlState.seen_sql_hashes。"""

    @staticmethod
    def has(state: TextToSqlState, fingerprint: str) -> bool:
        return fingerprint in set(state.get("seen_sql_hashes", []))

    @staticmethod
    def remember(state: TextToSqlState, fingerprint: str) -> list[str]:
        hashes = list(state.get("seen_sql_hashes", []))
        if fingerprint not in hashes:
            hashes.append(fingerprint)
        return hashes


def sql_fingerprint(sql: str, params: dict[str, Any] | None) -> str:
    payload = json.dumps(
        {"sql": sql.strip(), "params": params or {}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def max_repeat_same_error() -> int:
    return max(1, int(settings.FINANCIAL_SQL_MAX_REPEAT_SAME_ERROR))


def _next_repeat_count(state: TextToSqlState, error_type: ValidationErrorType) -> int:
    if not error_type:
        return 0
    last = str(state.get("last_error_type", ""))
    current = int(state.get("repeat_error_count", 0))
    if error_type == last:
        return current + 1
    return 1


def _abort_reason(
    state: TextToSqlState,
    *,
    error_type: ValidationErrorType,
    sql: str,
    params: dict[str, Any],
) -> str:
    attempts = int(state.get("attempts", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if attempts >= max_attempts:
        return "max_attempts_exceeded"

    fingerprint = sql_fingerprint(sql, params)
    if StateSqlRetryGuardStore.has(state, fingerprint):
        return "duplicate_sql"

    repeat_count = _next_repeat_count(state, error_type)
    if error_type and repeat_count >= max_repeat_same_error():
        return f"repeat_error:{error_type}"

    return ""


def resolve_retry_action(
    state: TextToSqlState,
    *,
    error_type: ValidationErrorType,
    error: str,
    sql: str,
    params: dict[str, Any],
    terminal_on_abort: RetryTerminal,
) -> dict[str, Any]:
    """决定走 correct_sql 还是提前结束纠错环。"""
    reason = _abort_reason(state, error_type=error_type, sql=sql, params=params)
    repeat_count = _next_repeat_count(state, error_type)
    fingerprint = sql_fingerprint(sql, params)

    if reason:
        message = error or reason
        if reason.startswith("repeat_error:"):
            message = f"{error}（连续相同错误 {repeat_count} 次，提前结束纠错）".strip()
        elif reason == "duplicate_sql":
            message = f"{error}（重复 SQL 已尝试过，提前结束纠错）".strip()

        return {
            "last_error_type": error_type,
            "repeat_error_count": repeat_count,
            "validation_error": message,
            "validation_error_type": error_type,
            "validation_errors": [message],
            "validation_error_types": [error_type] if error_type else [],
            "execution_error": message if terminal_on_abort == "execution_error_output" else "",
            "next_step": terminal_on_abort,
        }

    return {
        "last_error_type": error_type,
        "repeat_error_count": repeat_count,
        "seen_sql_hashes": StateSqlRetryGuardStore.remember(state, fingerprint),
        "validation_error": error,
        "validation_error_type": error_type,
        "validation_errors": [error] if error else [],
        "validation_error_types": [error_type] if error_type else [],
        "next_step": "correct_sql",
    }


__all__ = [
    "SqlRetryGuardStore",
    "StateSqlRetryGuardStore",
    "max_repeat_same_error",
    "resolve_retry_action",
    "sql_fingerprint",
]
