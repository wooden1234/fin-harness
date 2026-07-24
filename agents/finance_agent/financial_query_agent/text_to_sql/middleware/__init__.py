"""text_to_sql 中间件导出。"""

from .base import MiddlewareChain, MiddlewareResult, TextToSqlMiddleware, halt_updates
from .capability import DatabaseCapabilityMiddleware
from .clarification import ClarificationMiddleware


def default_middleware_chain() -> MiddlewareChain:
    return MiddlewareChain(
        [
            DatabaseCapabilityMiddleware(),
            ClarificationMiddleware(),
        ]
    )


__all__ = [
    "ClarificationMiddleware",
    "DatabaseCapabilityMiddleware",
    "MiddlewareChain",
    "MiddlewareResult",
    "TextToSqlMiddleware",
    "default_middleware_chain",
    "halt_updates",
]
