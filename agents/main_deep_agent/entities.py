"""Evidence 归属校验使用的实体别名与官方来源注册表。"""

from __future__ import annotations

import re


_ENTITY_ALIASES = {
    "Apple": ("Apple", "苹果", "苹果公司", "AAPL"),
    "Amazon": ("Amazon", "亚马逊", "亚马逊公司", "AMZN"),
    "Tencent": ("Tencent", "腾讯", "腾讯控股", "0700", "HK0700", "TCEHY"),
}

OFFICIAL_ENTITY_DOMAINS = {
    "Apple": ("apple.com",),
    "Amazon": ("amazon.com", "aboutamazon.com"),
    "Tencent": ("tencent.com", "tencent.com.cn"),
}


def normalize_entity(value: str) -> str:
    """归一已登记别名；未知实体保持 Agent 提供的原始表达。"""
    normalized = re.sub(r"\s+", "", value).lower()
    for canonical, aliases in _ENTITY_ALIASES.items():
        if normalized in {re.sub(r"\s+", "", item).lower() for item in aliases}:
            return canonical
    return value.strip()


def entity_aliases(entity: str) -> tuple[str, ...]:
    """返回 Evidence 文本匹配所需别名，未知实体仅使用自身。"""
    canonical = normalize_entity(entity)
    return _ENTITY_ALIASES.get(canonical, (canonical,))


__all__ = ["OFFICIAL_ENTITY_DOMAINS", "entity_aliases", "normalize_entity"]
