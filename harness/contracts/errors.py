"""Harness 运行时错误。"""


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


class ApprovalError(HarnessError):
    def __init__(self, message: str, *, code: str = "approval") -> None:
        super().__init__(message, code=code)
