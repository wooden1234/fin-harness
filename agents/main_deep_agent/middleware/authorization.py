"""工具权限与 Web 来源边界策略。"""

from __future__ import annotations

from urllib.parse import urlparse

from agents.runtime_context import AgentRuntimeContext


def normalize_entity(value: str) -> str:
    """规范化实体字符串；不做公司别名白名单映射。"""
    return str(value or "").strip()


def permissions_allow(context: AgentRuntimeContext, tool_id: str) -> bool:
    """仅允许运行时身份明确授权的工具。"""
    return "*" in context.permissions or tool_id in context.permissions


def normalize_search_domain(value: str) -> str:
    """将域名约束为不含协议、路径和 www 前缀的主机名。"""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").removeprefix("www.")


def domain_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def official_search_domains(entity: str) -> list[str]:
    """无公司域名白名单时，仅返回通用监管披露域名。"""
    del entity
    return ["sec.gov"]


def apply_source_preference(
    *,
    source_preference: str,
    entity: str,
    requested_domains: list[str],
) -> list[str]:
    """把来源偏好转换为可执行的 Web 域名边界。"""
    requested = list(
        dict.fromkeys(
            domain
            for domain in (
                normalize_search_domain(item) for item in requested_domains
            )
            if domain
        )
    )
    official = official_search_domains(entity)
    if source_preference == "official_first" and official:
        return official
    if source_preference == "independent_secondary" and requested:
        return [
            domain
            for domain in requested
            if not any(
                domain_matches(domain, official_domain)
                for official_domain in official
            )
        ]
    return requested


__all__ = [
    "apply_source_preference",
    "domain_matches",
    "normalize_entity",
    "normalize_search_domain",
    "official_search_domains",
    "permissions_allow",
]
