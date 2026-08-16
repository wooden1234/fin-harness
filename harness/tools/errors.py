"""工具错误码。"""

MALFORMED_ARGUMENTS = "malformed_arguments"


def error_result(code: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"ok": False, "error": code}
    payload.update(extra)
    return payload
