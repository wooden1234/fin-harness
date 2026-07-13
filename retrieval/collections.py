"""PGVector 集合名与 knowledge category 映射（可通过 .env 覆盖）。"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings

# category → (env 变量名, 默认 LlamaIndex table_name；实际 PG 表名为 data_{table_name})
_COLLECTION_DEFAULTS: dict[str, tuple[str, str]] = {
    "faq": ("PGVECTOR_COLLECTION_FAQ", "faq_md_vectors"),
    "macro_research": ("PGVECTOR_COLLECTION_MACRO", "fin_macro_vectors"),
    "annual_reports": (
        "PGVECTOR_COLLECTION_ANNUAL_REPORT",
        "fin_annual_report_vectors",
    ),
    "research_reports": (
        "PGVECTOR_COLLECTION_RESEARCH_REPORT",
        "fin_research_report_vectors",
    ),
    "industry_whitepapers": (
        "PGVECTOR_COLLECTION_INDUSTRY_WHITEPAPER",
        "fin_industry_whitepaper_vectors",
    ),
    "policy": ("PGVECTOR_COLLECTION_POLICY", "fin_policy_vectors"),
}


@lru_cache(maxsize=1)
def get_collection_registry() -> dict[str, str]:
    """category → pgvector 表名。"""
    registry: dict[str, str] = {}
    for category, (env_key, default) in _COLLECTION_DEFAULTS.items():
        if category == "faq":
            table = (
                settings.PGVECTOR_COLLECTION_FAQ
                or settings.PGVECTOR_TABLE_NAME
                or default
            )
        else:
            table = str(getattr(settings, env_key, default) or default)
        registry[category] = table
    return registry


def get_table_name(category: str) -> str:
    registry = get_collection_registry()
    if category not in registry:
        known = ", ".join(sorted(registry))
        raise KeyError(f"未知 category={category!r}，可选: {known}")
    return registry[category]


def all_categories() -> list[str]:
    return list(get_collection_registry().keys())


def pdf_categories() -> list[str]:
    return [c for c in all_categories() if c != "faq"]
