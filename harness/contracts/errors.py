"""Harness 运行时错误。"""

from __future__ import annotations


class HarnessError(Exception):
    def __init__(self, message: str = "", *, code: str = "error") -> None:
        super().__init__(message or code)
        self.code = code


class SessionFormatError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="session_format")


class LlmError(HarnessError):
    def __init__(self, message: str = "", *, code: str = "unknown", retryable: bool = False) -> None:
        super().__init__(message, code=code)
        self.retryable = retryable


class InvariantError(HarnessError):
    def __init__(self, message: str, *, code: str = "invariant") -> None:
        super().__init__(message, code=code)


class AgentBusyError(HarnessError):
    def __init__(self, message: str = "session busy") -> None:
        super().__init__(message, code="busy")


class PersistFailedError(HarnessError):
    def __init__(self, message: str = "persist failed") -> None:
        super().__init__(message, code="persist_failed")


class ContextBudgetExhaustedError(HarnessError):
    def __init__(
        self,
        *,
        context_window: int,
        final_tokens: int,
        stages: list[str],
        protected_tokens: int = 0,
    ) -> None:
        self.context_window = context_window
        self.final_tokens = final_tokens
        self.stages = tuple(stages)
        self.protected_tokens = protected_tokens
        super().__init__(
            f"context budget exhausted: {final_tokens}/{context_window}",
            code="context_budget_exhausted",
        )


class ApprovalError(HarnessError):
    def __init__(self, message: str, *, code: str = "approval") -> None:
        super().__init__(message, code=code)
