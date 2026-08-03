"""联网搜索域名边界：仅 allowlist / open 两种模式。"""

from __future__ import annotations

from urllib.parse import urlparse

# 默认金融白名单（allowlist 模式最多取前 5 个）。
DEFAULT_WEB_SEARCH_DOMAINS: tuple[str, ...] = (
    "cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
    "sec.gov",
    "10jqka.com.cn",
    "eastmoney.com",
    "stcn.com",
    "cs.com.cn",
    "yicai.com",
    "cls.cn",
    "reuters.com",
    "bloomberg.com",
)


def normalize_search_domain(value: str) -> str:
    """将域名约束为不含协议、路径和 www 前缀的主机名。"""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").removeprefix("www.")


def parse_domain_csv(raw: str) -> list[str]:
    return list(
        dict.fromkeys(
            domain
            for domain in (
                normalize_search_domain(item)
                for item in str(raw or "").split(",")
            )
            if domain
        )
    )


def resolve_search_domains(
    *,
    scope: str = "allowlist",
    allowed_domains: str = "",
    max_domains: int = 5,
) -> list[str]:
    """解析搜索域名。

    - allowlist: 默认金融白名单（或配置覆盖），最多 max_domains 个
    - open: 不限域名，返回空列表
    """
    mode = str(scope or "allowlist").strip().lower()
    if mode == "open":
        return []

    configured = parse_domain_csv(allowed_domains)
    base = configured or list(DEFAULT_WEB_SEARCH_DOMAINS)
    return base[: max(1, int(max_domains))]


__all__ = [
    "DEFAULT_WEB_SEARCH_DOMAINS",
    "normalize_search_domain",
    "parse_domain_csv",
    "resolve_search_domains",
]
