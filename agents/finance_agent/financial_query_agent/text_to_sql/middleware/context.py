"""text_to_sql 上下文注入中间件。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.text_to_sql.generation.context import (
    build_fewshot_examples,
    build_schema_prompt,
)
from agents.finance_agent.financial_query_agent.text_to_sql.middleware.base import (
    MiddlewareResult,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlState,
)


class ContextMiddleware:
    """在生成前注入 schema 与 few-shot 示例。"""

    async def before_generate(
        self,
        state: TextToSqlState,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        del config
        if state.get("schema_prompt") and state.get("fewshot_examples"):
            return None
        return MiddlewareResult(
            state_updates={
                "schema_prompt": build_schema_prompt(),
                "fewshot_examples": build_fewshot_examples(state["question"]),
            }
        )

    async def after_generate(self, state, generated, config=None) -> MiddlewareResult | None:
        del state, generated, config
        return None

    async def after_correct(self, state, corrected, config=None) -> MiddlewareResult | None:
        del state, corrected, config
        return None


__all__ = ["ContextMiddleware"]
