"""text_to_sql 中间件基础协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlState,
)


@dataclass
class MiddlewareResult:
    halt: bool = False
    halt_reason: str = ""
    halt_answer: str = ""
    state_updates: dict[str, Any] = field(default_factory=dict)


class TextToSqlMiddleware(Protocol):
    async def before_generate(
        self,
        state: TextToSqlState,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None: ...

    async def after_generate(
        self,
        state: TextToSqlState,
        generated: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None: ...

    async def after_correct(
        self,
        state: TextToSqlState,
        corrected: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None: ...


class MiddlewareChain:
    def __init__(self, middlewares: list[TextToSqlMiddleware]) -> None:
        self.middlewares = middlewares

    async def run_before_generate(
        self,
        state: TextToSqlState,
        config: RunnableConfig | None = None,
    ) -> tuple[TextToSqlState, MiddlewareResult | None]:
        current = dict(state)
        for middleware in self.middlewares:
            result = await middleware.before_generate(current, config)
            if not result:
                continue
            current.update(result.state_updates)
            if result.halt:
                return current, result
        return current, None

    async def run_after_generate(
        self,
        state: TextToSqlState,
        generated: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        for middleware in self.middlewares:
            result = await middleware.after_generate(state, generated, config)
            if result:
                return result
        return None

    async def run_after_correct(
        self,
        state: TextToSqlState,
        corrected: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        for middleware in self.middlewares:
            result = await middleware.after_correct(state, corrected, config)
            if result:
                return result
        return None


def halt_updates(result: MiddlewareResult) -> dict[str, Any]:
    return {
        **result.state_updates,
        "halted": True,
        "halt_reason": result.halt_reason,
        "halt_answer": result.halt_answer,
    }


__all__ = [
    "MiddlewareChain",
    "MiddlewareResult",
    "TextToSqlMiddleware",
    "halt_updates",
]
