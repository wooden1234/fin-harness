"""Evidence coverage, provenance, and source-lock evaluation."""

from __future__ import annotations

from typing import Any


def _locator(item: dict[str, Any]) -> str:
    return str(
        item.get("node_id")
        or item.get("evidence_id")
        or item.get("url")
        or item.get("doc_id")
        or item.get("snippet")
        or ""
    ).strip()


def _source(item: dict[str, Any]) -> str:
    return str(
        item.get("source_type")
        or item.get("source")
        or item.get("provider")
        or item.get("tool")
        or ""
    ).strip().casefold()


def _source_allowed(source: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    aliases = {
        "local_pdf": ("pdf", "knowledge.pdf"),
        "local_company_filing": ("annual_report", "filing", "pdf"),
        "company_filing": ("annual_report", "filing", "pdf"),
        "normalized_financial_database": ("finance", "iwencai"),
        "authoritative_financial_database": ("finance", "iwencai"),
        "central_bank_report": ("macro", "central_bank", "pdf"),
        "research_report": ("research_report", "report", "pdf"),
        "company_announcement": ("announcement",),
        "exchange": ("exchange", "sse", "szse", "hkex"),
        "regulator": ("regulator", "csrc", "pbc"),
        "government": ("government", "gov"),
    }
    lowered = source.casefold()
    return any(
        token in lowered
        for item in allowed
        for token in (item.casefold(), *aliases.get(item, ()))
    )


def evaluate_evidence(
    outputs: dict[str, Any], reference_outputs: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = dict(reference_outputs.get("evidence") or {})
    routing = dict(reference_outputs.get("routing_contract") or {})
    citations = [item for item in outputs.get("citations") or [] if isinstance(item, dict)]
    links = [item for item in outputs.get("claim_evidence_links") or [] if isinstance(item, dict)]
    facts = [item for item in reference_outputs.get("required_facts") or [] if isinstance(item, dict)]
    expected_gap = bool(expected.get("expected_gap"))
    required = bool(expected.get("required")) and not expected_gap
    valid = [item for item in citations if _locator(item)]
    evidence_ids = {
        str(item.get("evidence_id") or item.get("node_id") or "")
        for item in valid
        if item.get("evidence_id") or item.get("node_id")
    }
    linked_ids = {
        str(item.get("evidence_id") or "")
        for item in links
        if item.get("evidence_id")
    }

    results: list[dict[str, Any]] = [
        {
            "key": "evidence_provenance",
            "score": (len(valid) / len(citations)) if citations else (0.0 if required else None),
            "comment": f"citations={len(citations)} with_locator={len(valid)}",
        }
    ]
    if required:
        covered_ids = {
            str(item.get("claim_id") or item.get("fact_id") or "")
            for item in links
            if item.get("evidence_id")
        }
        if facts and links:
            coverage = min(1.0, len(covered_ids) / len(facts))
        else:
            coverage = 1.0 if valid else 0.0
        results.append(
            {
                "key": "evidence_coverage",
                "score": coverage,
                "comment": f"required_facts={len(facts)} linked={len(covered_ids)}",
            }
        )
    else:
        results.append(
            {
                "key": "evidence_coverage",
                "score": None,
                "comment": "Evidence is not required for this case.",
            }
        )

    if links:
        supported = linked_ids & evidence_ids
        support_score = len(supported) / len(linked_ids) if linked_ids else 0.0
    else:
        support_score = None
    results.append(
        {
            "key": "evidence_support",
            "score": support_score,
            "comment": f"linked_evidence={sorted(linked_ids)} observed_evidence={sorted(evidence_ids)}",
        }
    )

    invalid_citations = len(citations) - len(valid)
    dangling_links = sorted(linked_ids - evidence_ids) if evidence_ids else sorted(linked_ids)
    if invalid_citations or dangling_links:
        results.append(
            {
                "key": "citation_hard_failure",
                "score": 0.0,
                "comment": (
                    f"citations_without_locator={invalid_citations} "
                    f"dangling_evidence_links={dangling_links}"
                ),
            }
        )

    preferred = [str(item) for item in expected.get("preferred_doc_ids") or []]
    cited_docs = {
        str(item.get("doc_id") or "")
        for item in citations
        if str(item.get("doc_id") or "")
    }
    results.append(
        {
            "key": "preferred_source_match",
            "score": (
                len(set(preferred) & cited_docs) / len(set(preferred))
                if preferred
                else None
            ),
            "comment": "Preferred sources are diagnostic and never imply source lock.",
        }
    )

    if routing.get("source_policy") == "source_locked":
        allowed = [str(item) for item in routing.get("allowed_source_types") or []]
        observed = [_source(item) for item in citations if _source(item)]
        compliant = bool(valid) and (
            not allowed
            or (bool(observed) and all(_source_allowed(item, allowed) for item in observed))
        )
        if expected_gap:
            compliant = not citations
        results.append(
            {
                "key": "source_lock_hard_failure",
                "score": 1.0 if compliant else 0.0,
                "comment": f"allowed={allowed} observed={observed} expected_gap={expected_gap}",
            }
        )
    return results


__all__ = ["evaluate_evidence"]
