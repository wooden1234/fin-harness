"""Map concrete tool calls to stable business capability domains."""

from __future__ import annotations

from typing import Any


_TOOL_CAPABILITIES = {
    "query_iwencai_finance": "financial_metric",
    "lookup_financial_fact": "financial_metric",
    "lookup_knowledge_fact": "financial_metric",
    "search_pdf_knowledge_tool": "local_document_research",
    "catalog_pdf_knowledge_tool": "local_document_catalog",
    "search_iwencai_announcement": "announcement_search",
    "search_iwencai_report": "research_report_search",
    "query_iwencai_rating": "institution_rating",
    "screen_iwencai": "astock_screening",
    "screen_iwencai_fund": "fund_screening",
    "screen_iwencai_usstock": "usstock_screening",
    "query_iwencai_market": "security_market_data",
    "query_iwencai_index": "index_market_data",
    "query_iwencai_industry": "industry_market_data",
    "search_iwencai_news": "financial_news",
    "search_web": "open_web",
    "run_calculation": "calculation",
    "finalign_analyze": "synthesis",
    "memory_write": "preference_memory",
    "memory_delete": "preference_delete",
    "tyc_company_registration": "company_registration",
}

_INTENT_ALLOWED = {
    "financial_metric": {"financial_metric"},
    "company_operating_metric": {"financial_metric", "local_document_research", "announcement_search"},
    "macro_indicator": {"local_document_research", "open_web"},
    "filing_detail": {"financial_metric", "local_document_research"},
    "quarterly_financial_metric": {"financial_metric", "local_document_research"},
    "segment_financial_metric": {"financial_metric", "local_document_research"},
    "astock_screening": {"astock_screening"},
    "fund_screening": {"fund_screening"},
    "company_registration": {"company_registration", "local_document_research"},
    "announcement_search": {"announcement_search"},
    "institution_rating": {"institution_rating"},
    "research_report_search": {"research_report_search"},
    "financial_comparison_calculation": {"financial_metric"},
    "financial_comparison": {"financial_metric"},
    "financial_reconciliation": {"financial_metric", "local_document_research"},
    "cross_filing_comparison": {"local_document_research"},
    "segment_financial_calculation": {"financial_metric", "local_document_research"},
    "macro_calculation": {"local_document_research", "open_web"},
    "company_financial_analysis": {"financial_metric", "local_document_research", "research_report_search"},
    "filing_analysis": {"local_document_research"},
    "filing_document_research": {"local_document_research", "announcement_search"},
    "research_report_analysis": {"local_document_research", "research_report_search"},
    "macro_report_analysis": {"local_document_research"},
    "local_filing_research": {"local_document_research"},
    "source_locked_document_research": {"local_document_research"},
    "local_document_research": {"local_document_research"},
    "industry_market_data_and_news": {"industry_market_data", "financial_news"},
    "index_market_data": {"index_market_data"},
    "security_market_data": {"security_market_data"},
    "policy_research": {"open_web", "financial_news", "local_document_research"},
}


def tool_capability(name: str) -> str:
    normalized = str(name or "").strip()
    if normalized in _TOOL_CAPABILITIES:
        return _TOOL_CAPABILITIES[normalized]
    lowered = normalized.casefold()
    if lowered.startswith("tyc.") or "registration" in lowered:
        return "company_registration"
    if lowered in {"skill", "todo_write", "todo_read"}:
        return "control"
    return "unknown"


def evaluate_routing(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    contract = dict(reference_outputs.get("routing_contract") or {})
    intent = str(contract.get("intent_domain") or "")
    names = [
        str(item.get("name") or "")
        for item in outputs.get("tool_calls") or []
        if isinstance(item, dict)
    ]
    capabilities = [tool_capability(name) for name in names]
    material = {item for item in capabilities if item not in {"control", "synthesis", "calculation"}}
    policy = str(contract.get("source_policy") or "default")
    if policy == "no_external_fact":
        passed = not material
    else:
        allowed = _INTENT_ALLOWED.get(intent)
        passed = bool(material & allowed) if allowed is not None else bool(material or names)
        if intent == "industry_market_data_and_news":
            passed = {"industry_market_data", "financial_news"}.issubset(material)
        if intent.startswith("clarification") or intent in {"secret_protection", "citation_integrity"}:
            passed = not material

    forbidden = {str(item) for item in contract.get("forbidden_domains") or []}
    forbidden_hit = sorted(material & forbidden)
    if "all_external_tools" in forbidden and material:
        forbidden_hit = sorted(material)
    results = [
        {
            "key": "route_capability_match",
            "score": 1.0 if passed else 0.0,
            "comment": f"intent={intent} tools={names} capabilities={capabilities}",
        }
    ]
    if forbidden_hit:
        results.append(
            {
                "key": "route_hard_failure",
                "score": 0.0,
                "comment": f"forbidden capabilities used: {forbidden_hit}",
            }
        )
    return results


__all__ = ["evaluate_routing", "tool_capability"]
