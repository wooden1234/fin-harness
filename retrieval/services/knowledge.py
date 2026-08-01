"""Direct Agent 与 DeepAgent Tool 共用的知识检索和准入服务。"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

from retrieval import RetrievalHit, get_faq_retriever, get_pdf_retriever

FaqDomain = Literal["capital_market", "corporate_finance"]
PdfCategory = Literal[
    "annual_reports",
    "research_reports",
    "industry_whitepapers",
    "macro_research",
    "policy",
]

_ROOT = Path(__file__).resolve().parents[2]
_PDF_MANIFEST = _ROOT / "knowledge" / "raw" / "manifest_pdf.yaml"
_PDF_CATEGORIES = frozenset(
    {
        "annual_reports",
        "research_reports",
        "industry_whitepapers",
        "macro_research",
        "policy",
    }
)
_CORPORATE_MARKERS = (
    "报销",
    "发票",
    "付款",
    "审批",
    "预算",
    "内控",
    "应收",
    "应付",
    "库存",
    "盘点",
    "月结",
    "关账",
    "备用金",
    "财务制度",
)
_CATALOG_ENTITY_ALIASES = {
    "TCEHY": ("腾讯", "腾讯控股"),
    "CATL": ("宁德时代",),
    "688256": ("寒武纪", "中科寒武纪"),
    "688047": ("龙芯中科",),
}

CORPORATE_TEMPLATE_NOTICE = "企业内部制度模板，仅供参考，具体执行以公司正式制度为准。"


def infer_faq_domain(query: str) -> FaqDomain:
    """仅在存在明确企业制度语义时启用 corporate_finance。"""
    return (
        "corporate_finance"
        if any(marker in str(query or "") for marker in _CORPORATE_MARKERS)
        else "capital_market"
    )


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _pdf_catalog() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(_PDF_MANIFEST.read_text(encoding="utf-8")) or {}
    category_defaults = raw.get("categories") or {}
    cleaning_version = str(raw.get("version") or "")
    catalog: dict[str, dict[str, Any]] = {}
    for category, documents in (raw.get("documents") or {}).items():
        if category not in _PDF_CATEGORIES:
            continue
        defaults = dict(category_defaults.get(category) or {})
        for document in documents or []:
            item = {**defaults, **dict(document)}
            item["category"] = category
            item["cleaning_version"] = str(
                item.get("cleaning_version") or cleaning_version
            )
            # 首期 policy 统一隔离；其他类别只按 manifest 的显式准入字段开放。
            default_quality = (
                "quarantined"
                if category == "policy"
                else "approved" if item.get("ingest") else "rejected"
            )
            item["quality_status"] = str(
                item.get("quality_status") or default_quality
            )
            item["use_modes"] = list(
                item.get("use_modes")
                or []
            )
            item["evidence_role"] = str(
                item.get("evidence_role")
                or {
                    "annual_reports": "primary_fact",
                    "macro_research": "primary_fact",
                    "research_reports": "secondary_analysis",
                    "industry_whitepapers": "contextual_view",
                    "policy": "primary_fact",
                }.get(category, "contextual_view")
            )
            item["publication_date"] = str(
                item.get("publication_date") or item.get("effective_date") or ""
            )
            doc_id = str(item.get("doc_id") or "").strip()
            if doc_id:
                catalog[doc_id] = item
    return catalog


def _document_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    aliases = list(item.get("aliases") or [])
    aliases.extend(_CATALOG_ENTITY_ALIASES.get(str(item.get("ticker") or ""), ()))
    return {
        "doc_id": item.get("doc_id", ""),
        "title": item.get("title", ""),
        "category": item.get("category", ""),
        "issuer": item.get("issuer", ""),
        "publication_date": item.get("publication_date", ""),
        "quality_status": item.get("quality_status", "quarantined"),
        "use_modes": list(item.get("use_modes") or []),
        "evidence_role": item.get("evidence_role", "contextual_view"),
        "ticker": item.get("ticker"),
        "fiscal_year": item.get("fiscal_year"),
        "aliases": list(dict.fromkeys(aliases)),
        "cleaning_version": item.get("cleaning_version", ""),
        "content_hash": item.get("content_hash", ""),
    }


def catalog_pdf_documents(
    *,
    query: str = "",
    entity: str = "",
    year: int | None = None,
    categories: Sequence[str] = (),
    doc_id: str = "",
    use_mode: str = "research_evidence",
) -> list[dict[str, Any]]:
    """查询已登记且通过指定使用模式准入的 PDF 描述。"""
    category_set = {str(item) for item in categories if str(item) in _PDF_CATEGORIES}
    query_term = query.strip().lower()
    entity_term = entity.strip().lower()
    results: list[dict[str, Any]] = []
    for item in _pdf_catalog().values():
        if doc_id and item.get("doc_id") != doc_id:
            continue
        if category_set and item.get("category") not in category_set:
            continue
        if item.get("quality_status") != "approved":
            continue
        if use_mode not in set(item.get("use_modes") or []):
            continue
        if year is not None and str(item.get("fiscal_year") or item.get("publication_date") or "")[:4] != str(year):
            continue
        aliases = [
            *list(item.get("aliases") or []),
            *_CATALOG_ENTITY_ALIASES.get(str(item.get("ticker") or ""), ()),
        ]
        identifiers = [
            str(item.get("doc_id") or ""),
            str(item.get("title") or ""),
            str(item.get("ticker") or ""),
            str(item.get("issuer") or ""),
            str(item.get("fiscal_year") or ""),
            *aliases,
        ]
        haystack = " ".join(identifiers).lower()
        if entity_term and entity_term not in haystack:
            continue
        if query_term and query_term not in haystack and not any(
            len(identifier.strip()) >= 2
            and identifier.strip().lower() in query_term
            for identifier in identifiers
        ):
            continue
        results.append(_document_descriptor(item))
    return results


async def search_faq_knowledge(
    query: str,
    *,
    domain: FaqDomain,
    top_k: int = 3,
) -> list[RetrievalHit]:
    """检索指定 FAQ 域；企业模板必须由调用方显式选择。"""
    k = min(max(int(top_k), 1), 5)
    retriever = get_faq_retriever(
        top_k=k,
        similarity_threshold=None,
        metadata_filters={"domain": domain},
    )
    return await retriever.asearch(
        query,
        top_k=k,
        metadata_filters={"domain": domain},
    )


def _admitted_pdf_hit(
    hit: RetrievalHit,
    *,
    use_mode: str,
    allowed_doc_ids: set[str],
) -> bool:
    doc_id = str(hit.metadata.get("doc_id") or "").strip()
    item = _pdf_catalog().get(doc_id)
    if item is None or item.get("quality_status") != "approved":
        return False
    if use_mode not in set(item.get("use_modes") or []):
        return False
    return not allowed_doc_ids or doc_id in allowed_doc_ids


def admit_pdf_hits(
    hits: Sequence[RetrievalHit],
    *,
    use_mode: str,
    doc_ids: Sequence[str] = (),
) -> list[RetrievalHit]:
    """对任意 PDF 检索结果执行统一 manifest 准入。"""
    allowed_doc_ids = {str(item) for item in doc_ids if str(item)}
    return [
        hit
        for hit in hits
        if _admitted_pdf_hit(
            hit,
            use_mode=use_mode,
            allowed_doc_ids=allowed_doc_ids,
        )
    ]


async def search_pdf_knowledge(
    query: str,
    *,
    categories: Sequence[str],
    doc_ids: Sequence[str] = (),
    top_k: int = 5,
    use_mode: str = "research_evidence",
    metadata_filters: dict[str, Any] | None = None,
) -> list[RetrievalHit]:
    """检索 PDF 并以 manifest 再做一次不可绕过的质量准入。"""
    selected_categories = [
        str(item) for item in categories if str(item) in _PDF_CATEGORIES
    ]
    if not selected_categories:
        raise ValueError("pdf_categories_required")
    allowed_doc_ids = {str(item) for item in doc_ids if str(item)}
    filters = dict(metadata_filters or {})
    if len(allowed_doc_ids) == 1:
        filters["doc_id"] = next(iter(allowed_doc_ids))
    k = min(max(int(top_k), 1), 8)
    retriever = get_pdf_retriever(
        categories=selected_categories,
        top_k=k,
        similarity_threshold=None,
        metadata_filters=filters or None,
        hybrid=True,
    )
    hits = await retriever.asearch(query, top_k=k, metadata_filters=filters or None)
    return admit_pdf_hits(hits, use_mode=use_mode, doc_ids=allowed_doc_ids)


def evidence_from_hit(
    hit: RetrievalHit,
    *,
    research_question_id: str,
    source_type: str,
    task_id: str = "deep-research",
) -> dict[str, Any]:
    """把检索命中转换为可直接通过 AgentResult 校验的 Evidence。"""
    metadata = dict(hit.metadata or {})
    doc_id = str(metadata.get("doc_id") or metadata.get("source") or "unknown")
    page = metadata.get("page_num") or metadata.get("page")
    section = metadata.get("section_path") or metadata.get("section") or ""
    content_hash = _hash_text(hit.text or "")
    digest = hashlib.sha256(
        f"{doc_id}:{page}:{section}:{content_hash}".encode("utf-8")
    ).hexdigest()[:16]
    category = str(metadata.get("category") or "faq")
    catalog_item = _pdf_catalog().get(doc_id, {})
    authority = str(
        metadata.get("authority_tier")
        or catalog_item.get("authority_tier")
        or "reference"
    )
    evidence_role = str(
        metadata.get("evidence_role")
        or catalog_item.get("evidence_role")
        or ("contextual_view" if category == "faq" else "secondary_analysis")
    )
    claim_type = "disclosed_view" if evidence_role != "primary_fact" else "fact"
    return {
        "evidence_id": f"knowledge:{digest}",
        "task_id": task_id,
        "source_type": source_type,
        "provider": "local_knowledge",
        "title": str(metadata.get("title") or catalog_item.get("title") or doc_id),
        "content": str(hit.text or ""),
        "published_at": str(
            metadata.get("publication_date")
            or metadata.get("effective_date")
            or catalog_item.get("publication_date")
            or ""
        ) or None,
        "confidence": min(max(float(hit.score), 0.0), 1.0),
        "metadata": {
            "research_question_id": research_question_id,
            "doc_id": doc_id,
            "domain": metadata.get("domain"),
            "category": category,
            "authority_tier": authority,
            "quality_status": catalog_item.get("quality_status", "approved"),
            "evidence_role": evidence_role,
            "claim_type": claim_type,
            "source_group": doc_id,
            "page": page,
            "section": section,
            "content_hash": content_hash,
            "template_notice": (
                CORPORATE_TEMPLATE_NOTICE
                if metadata.get("domain") == "corporate_finance"
                else ""
            ),
        },
    }


__all__ = [
    "CORPORATE_TEMPLATE_NOTICE",
    "admit_pdf_hits",
    "catalog_pdf_documents",
    "evidence_from_hit",
    "infer_faq_domain",
    "search_faq_knowledge",
    "search_pdf_knowledge",
]
