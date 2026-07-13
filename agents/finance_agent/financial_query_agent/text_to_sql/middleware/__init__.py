"""text_to_sql 中间件导出。"""

from .base import MiddlewareChain, MiddlewareResult, TextToSqlMiddleware, halt_updates
from .clarification import ClarificationMiddleware
from .context import ContextMiddleware


def default_middleware_chain() -> MiddlewareChain:
    return MiddlewareChain(
        [
            ClarificationMiddleware(),
            ContextMiddleware(),
        ]
    )


__all__ = [
    "ClarificationMiddleware",
    "ContextMiddleware",
    "MiddlewareChain",
    "MiddlewareResult",
    "TextToSqlMiddleware",
    "default_middleware_chain",
    "halt_updates",
]
