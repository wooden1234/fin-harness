"""Elasticsearch 客户端与索引命名工具。"""

from __future__ import annotations

from typing import Any

from app.core.config import settings

ES_INDEX_ENV_BY_CATEGORY = {
    "faq": "ELASTICSEARCH_INDEX_FAQ",
    "macro_research": "ELASTICSEARCH_INDEX_MACRO",
    "annual_reports": "ELASTICSEARCH_INDEX_ANNUAL_REPORT",
    "research_reports": "ELASTICSEARCH_INDEX_RESEARCH_REPORT",
    "industry_whitepapers": "ELASTICSEARCH_INDEX_INDUSTRY_WHITEPAPER",
    "policy": "ELASTICSEARCH_INDEX_POLICY",
}


def create_es_client() -> Any:
    from elasticsearch import Elasticsearch

    if not settings.ELASTICSEARCH_URL:
        raise RuntimeError("未配置 ELASTICSEARCH_URL，无法连接 Elasticsearch")
    if settings.ELASTICSEARCH_USERNAME or settings.ELASTICSEARCH_PASSWORD:
        return Elasticsearch(
            settings.ELASTICSEARCH_URL,
            basic_auth=(
                settings.ELASTICSEARCH_USERNAME,
                settings.ELASTICSEARCH_PASSWORD,
            ),
        )
    return Elasticsearch(settings.ELASTICSEARCH_URL)


def index_name(category: str) -> str:
    env_name = ES_INDEX_ENV_BY_CATEGORY.get(category)
    if env_name:
        configured = str(getattr(settings, env_name, "") or "")
        if configured:
            return configured
    prefix = settings.ELASTICSEARCH_INDEX_PREFIX or "fin_agent"
    return f"{prefix}_{category}"
