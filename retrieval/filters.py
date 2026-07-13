"""Metadata filter helpers for PDF retrieval."""

from __future__ import annotations

import re
from typing import Any

MetadataFilters = dict[str, Any]

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_TICKER_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "CATL": ("CATL", "宁德时代", "寧德時代"),
    "TCEHY": ("TCEHY", "Tencent", "腾讯", "騰訊"),
    "688256": ("688256", "Cambricon", "寒武纪", "寒武紀"),
    "688047": ("688047", "Loongson", "龙芯中科", "龍芯中科"),
}

CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("annual_reports", ("年报", "年度报告", "annual report", "财报", "財報")),
    ("research_reports", ("研报", "研究报告", "证券", "券商", "research report")),
    ("industry_whitepapers", ("白皮书", "whitepaper", "white paper")),
    ("policy", ("政策", "规划", "行动计划", "办法", "规则", "监管", "披露")),
    ("macro_research", ("宏观", "央行", "货币政策", "金融稳定", "区域金融")),
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def compact_filters(filters: MetadataFilters | None) -> MetadataFilters:
    """Drop empty values so callers can pass optional query params directly."""
    return {k: v for k, v in (filters or {}).items() if v not in (None, "", [], ())}


def infer_pdf_metadata_filters(query: str) -> MetadataFilters:
    """Infer coarse PDF metadata filters from a natural-language query."""
    text = _norm(query)
    filters: MetadataFilters = {}

    categories = [
        category
        for category, keywords in CATEGORY_KEYWORDS
        if any(_norm(keyword) in text for keyword in keywords)
    ]
    if categories:
        filters["category"] = categories[0] if len(categories) == 1 else categories

    year_match = _YEAR_RE.search(query)
    if year_match:
        filters["year"] = int(year_match.group(1))

    for company_key, aliases in COMPANY_ALIASES.items():
        if any(_norm(alias) in text for alias in aliases):
            filters["company"] = company_key
            break

    ticker_match = _TICKER_RE.search(query)
    if ticker_match:
        filters["ticker"] = ticker_match.group(1)
        filters.setdefault("company", ticker_match.group(1))

    doc_id_match = re.search(r"PDF-[A-Z0-9-]+", query, flags=re.IGNORECASE)
    if doc_id_match:
        filters["doc_id"] = doc_id_match.group(0).upper()

    return filters


def filter_categories(filters: MetadataFilters | None) -> list[str] | None:
    values = _as_list((filters or {}).get("category"))
    categories = [str(v) for v in values if v]
    return categories or None


def merge_filters(
    base: MetadataFilters | None,
    override: MetadataFilters | None,
) -> MetadataFilters:
    merged = dict(base or {})
    merged.update(compact_filters(override))
    return compact_filters(merged)


def metadata_matches(metadata: dict[str, Any], filters: MetadataFilters | None) -> bool:
    filters = compact_filters(filters)
    if not filters:
        return True

    categories = filter_categories(filters)
    if categories and _norm(metadata.get("category")) not in {_norm(c) for c in categories}:
        return False

    year = filters.get("year") or filters.get("fiscal_year")
    if year is not None and not _metadata_has_year(metadata, str(year)):
        return False

    company = filters.get("company")
    if company and not _metadata_has_company(metadata, str(company)):
        return False

    source = filters.get("source")
    if source and not _metadata_contains(
        metadata,
        str(source),
        fields=("source", "file", "title", "doc_id", "doc_group"),
    ):
        return False

    doc_id = filters.get("doc_id")
    if doc_id and _norm(doc_id) not in _norm(metadata.get("doc_id")):
        return False

    ticker = filters.get("ticker")
    if ticker and _norm(ticker) != _norm(metadata.get("ticker")):
        return False

    issuer = filters.get("issuer")
    if issuer and not _metadata_contains(metadata, str(issuer), fields=("issuer", "title")):
        return False

    return True


def _metadata_has_year(metadata: dict[str, Any], year: str) -> bool:
    if str(metadata.get("fiscal_year") or "") == year:
        return True
    effective_date = str(metadata.get("effective_date") or "")
    if effective_date.startswith(year):
        return True
    return _metadata_contains(
        metadata,
        year,
        fields=("source", "file", "title", "doc_id", "doc_group", "page_range"),
    )


def _metadata_has_company(metadata: dict[str, Any], company: str) -> bool:
    aliases = _company_aliases(company)
    haystack = " ".join(
        _norm(metadata.get(field))
        for field in ("ticker", "source", "file", "title", "doc_id", "doc_group", "issuer")
    )
    return any(_norm(alias) in haystack for alias in aliases)


def _company_aliases(company: str) -> tuple[str, ...]:
    normalized = _norm(company)
    for key, aliases in COMPANY_ALIASES.items():
        values = (key, *aliases)
        if normalized in {_norm(value) for value in values}:
            return values
    return (company,)


def _metadata_contains(
    metadata: dict[str, Any],
    needle: str,
    *,
    fields: tuple[str, ...],
) -> bool:
    value = _norm(needle)
    return any(value in _norm(metadata.get(field)) for field in fields)
