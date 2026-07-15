"""Milvus 客户端与 collection 命名工具。"""

from __future__ import annotations

from typing import Any

from app.core.config import settings

MILVUS_COLLECTION_ENV_BY_CATEGORY = {
    "faq": "MILVUS_COLLECTION_FAQ",
    "macro_research": "MILVUS_COLLECTION_MACRO",
    "annual_reports": "MILVUS_COLLECTION_ANNUAL_REPORT",
    "research_reports": "MILVUS_COLLECTION_RESEARCH_REPORT",
    "industry_whitepapers": "MILVUS_COLLECTION_INDUSTRY_WHITEPAPER",
    "policy": "MILVUS_COLLECTION_POLICY",
}


def milvus_enabled() -> bool:
    return bool(settings.MILVUS_ENABLED and settings.MILVUS_URI)


def create_milvus_client() -> Any:
    from pymilvus import MilvusClient

    if not settings.MILVUS_URI:
        raise RuntimeError("未配置 MILVUS_URI，无法连接 Milvus")

    kwargs: dict[str, Any] = {
        "uri": settings.MILVUS_URI,
    }
    if settings.MILVUS_TOKEN:
        kwargs["token"] = settings.MILVUS_TOKEN
    return MilvusClient(**kwargs)


def collection_name(category: str) -> str:
    env_name = MILVUS_COLLECTION_ENV_BY_CATEGORY.get(category)
    if env_name:
        configured = str(getattr(settings, env_name, "") or "")
        if configured:
            return configured
    prefix = settings.MILVUS_COLLECTION_PREFIX or "fin_agent"
    return f"{prefix}_{category}"
