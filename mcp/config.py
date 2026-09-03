"""MCP 运行时配置（从 .env / Settings 读取）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

_TYC_SERVER = "tyc-mcp"


def _settings() -> Settings | None:
    try:
        from app.core.config import settings

        return settings
    except Exception:  # noqa: BLE001
        return None


def normalize_bearer_token(token: str) -> str:
    """把裸 token 规范为 ``Bearer ...`` 请求头值。"""
    value = (token or "").strip()
    if not value:
        return ""
    if value.lower().startswith("bearer "):
        return value
    return f"Bearer {value}"


def tyc_mcp_url() -> str:
    settings = _settings()
    if settings is None:
        return ""
    return (settings.TYC_MCP_URL or "").strip()


def tyc_mcp_token() -> str:
    settings = _settings()
    if settings is None:
        return ""
    return (settings.TYC_MCP_TOKEN or "").strip()


def tyc_mcp_timeout_sec() -> float:
    settings = _settings()
    if settings is None:
        return 30.0
    return max(5.0, float(settings.TYC_MCP_TIMEOUT_SEC))


def tyc_mcp_configured() -> bool:
    settings = _settings()
    if settings is None or not settings.TYC_MCP_ENABLED:
        return False
    return bool(tyc_mcp_url() and tyc_mcp_token())


def tyc_mcp_auth_header() -> str:
    return normalize_bearer_token(tyc_mcp_token())


def resolve_server_config(server: str) -> dict[str, str] | None:
    """按逻辑 server 名解析 URL 与 Authorization。"""
    if server != _TYC_SERVER:
        return None
    if not tyc_mcp_configured():
        return None
    return {
        "url": tyc_mcp_url(),
        "authorization": tyc_mcp_auth_header(),
    }


__all__ = [
    "normalize_bearer_token",
    "resolve_server_config",
    "tyc_mcp_auth_header",
    "tyc_mcp_configured",
    "tyc_mcp_timeout_sec",
    "tyc_mcp_token",
    "tyc_mcp_url",
]
