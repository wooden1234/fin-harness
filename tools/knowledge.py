"""供 DeepAgent 使用的受治理本地知识检索工具。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool

from retrieval.services.knowledge import (
    CORPORATE_TEMPLATE_NOTICE,
    catalog_pdf_documents,
    evidence_from_hit,
    search_faq_knowledge,
    search_pdf_knowledge,
)
from tools.base import ToolSpec
from tools.registry import register_tool

PdfCategory = Literal[
    "annual_reports",
    "research_reports",
    "industry_whitepapers",
    "macro_research",
    "policy",
]


@tool(parse_docstring=True)
async def search_faq_knowledge_tool(
    query: str,
    research_question_id: str,
    domain: Literal["capital_market", "corporate_finance"],
    top_k: int = 3,
) -> dict[str, Any]:
    """检索稳定金融规则、概念或企业财务制度模板。

    Args:
        query: 独立、完整的检索问题
        research_question_id: ResearchPlan 中对应的问题 ID
        domain: 资本市场规则或企业财务模板域
        top_k: 返回条数，最大 5
    """
    hits = await search_faq_knowledge(query, domain=domain, top_k=top_k)
    evidence = [
        evidence_from_hit(
            hit,
            research_question_id=research_question_id,
            source_type="knowledge.faq.search",
        )
        for hit in hits
    ]
    return {
        "query": query,
        "domain": domain,
        "evidence": evidence,
        "template_notice": CORPORATE_TEMPLATE_NOTICE if domain == "corporate_finance" else "",
    }


@tool(parse_docstring=True)
async def catalog_pdf_knowledge_tool(
    query: str = "",
    entity: str = "",
    year: int | None = None,
    categories: list[PdfCategory] | None = None,
    doc_id: str = "",
) -> dict[str, Any]:
    """查询本地 PDF 目录，不读取正文。

    Args:
        query: 文档主题或名称关键词
        entity: 公司、机构或行业实体
        year: 报告年度或发布日期年份
        categories: 明确的文档类别
        doc_id: 已知文档 ID
    """
    documents = catalog_pdf_documents(
        query=query,
        entity=entity,
        year=year,
        categories=categories or (),
        doc_id=doc_id,
    )
    return {"documents": documents, "count": len(documents)}


@tool(parse_docstring=True)
async def search_pdf_knowledge_tool(
    query: str,
    research_question_id: str,
    categories: list[PdfCategory],
    doc_ids: list[str] | None = None,
    top_k: int = 5,
    source_locked: bool = False,
) -> dict[str, Any]:
    """检索已通过质量准入的本地 PDF 正文。

    Args:
        query: 独立、完整的文档检索问题
        research_question_id: ResearchPlan 中对应的问题 ID
        categories: 已通过目录判断的文档类别，禁止为空
        doc_ids: 目录返回的文档 ID；来源锁定时必须提供
        top_k: 返回条数，最大 8
        source_locked: 是否禁止检索指定文档以外的内容
    """
    selected_doc_ids = [str(item) for item in (doc_ids or []) if str(item)]
    if source_locked and not selected_doc_ids:
        return {"ok": False, "error": "source_locked_doc_ids_required"}
    hits = await search_pdf_knowledge(
        query,
        categories=categories,
        doc_ids=selected_doc_ids,
        top_k=top_k,
    )
    evidence = [
        evidence_from_hit(
            hit,
            research_question_id=research_question_id,
            source_type="knowledge.pdf.search",
        )
        for hit in hits
    ]
    return {"query": query, "evidence": evidence, "count": len(evidence)}


register_tool(
    ToolSpec(
        tool_id="knowledge.faq.search",
        name="search_faq_knowledge_tool",
        description="检索已分域的稳定金融 FAQ 和企业财务制度模板",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=search_faq_knowledge_tool,
)
register_tool(
    ToolSpec(
        tool_id="knowledge.pdf.catalog",
        name="catalog_pdf_knowledge_tool",
        description="查询通过质量准入的本地 PDF 文档目录",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=catalog_pdf_knowledge_tool,
)
register_tool(
    ToolSpec(
        tool_id="knowledge.pdf.search",
        name="search_pdf_knowledge_tool",
        description="检索通过质量准入的本地 PDF 原文证据",
        risk_level="low",
        read_only=True,
    ),
    langchain_tool=search_pdf_knowledge_tool,
)


__all__ = [
    "catalog_pdf_knowledge_tool",
    "search_faq_knowledge_tool",
    "search_pdf_knowledge_tool",
]
