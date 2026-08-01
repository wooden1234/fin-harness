"""在线和离线共用的确定性评测入口。"""

from __future__ import annotations

from typing import Any


def evaluate_run(result: dict[str, Any]) -> dict[str, Any]:
    """评估可从运行结果直接验证的路由、证据、降级和延迟约束。"""
    checks: list[dict[str, Any]] = []
    execution_status = str(result.get("execution_status") or "")
    answer = _answer_text(result)
    route = str(result.get("route") or "")

    _add_check(
        checks,
        "answer_available",
        bool(answer) or route == "clarify",
        actual={"route": route, "answer_length": len(answer)},
    )
    _add_check(
        checks,
        "terminal_status",
        execution_status not in {"hard_timeout", "failed"},
        actual=execution_status or "unspecified",
    )

    expected_route = result.get("expected_route")
    if expected_route is not None:
        _add_check(
            checks,
            "route_accuracy",
            route == str(expected_route),
            expected=str(expected_route),
            actual=route,
        )

    citations = [
        item for item in result.get("citations") or [] if isinstance(item, dict)
    ]
    citation_required = bool(result.get("citation_required"))
    if citations or citation_required:
        valid_citations = [item for item in citations if _citation_has_provenance(item)]
        _add_check(
            checks,
            "citation_provenance",
            bool(valid_citations) and len(valid_citations) == len(citations),
            actual={
                "total": len(citations),
                "with_provenance": len(valid_citations),
            },
        )

    claims = [item for item in result.get("claims") or [] if isinstance(item, dict)]
    links = [
        item
        for item in result.get("claim_evidence_links") or []
        if isinstance(item, dict)
    ]
    if claims:
        linked_claim_ids = {
            str(item.get("claim_id") or "")
            for item in links
            if item.get("evidence_id")
        }
        material_claim_ids = {
            str(item.get("claim_id") or "")
            for item in claims
            if str(item.get("claim_type") or "fact") != "opinion"
        }
        covered = material_claim_ids & linked_claim_ids
        _add_check(
            checks,
            "claim_evidence_coverage",
            not material_claim_ids or covered == material_claim_ids,
            actual={"material": len(material_claim_ids), "covered": len(covered)},
        )

    latency_ms = _number(result.get("latency_ms"))
    slo_ms = _number(result.get("slo_ms"))
    if latency_ms is not None and slo_ms is not None:
        _add_check(
            checks,
            "latency_slo",
            latency_ms <= slo_ms,
            expected={"max_ms": slo_ms},
            actual={"latency_ms": latency_ms},
        )

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "metrics": {
            "citation_count": len(citations),
            "answer_length": len(answer),
            **({"latency_ms": latency_ms} if latency_ms is not None else {}),
        },
        "result_keys": sorted(result.keys()),
    }


def _answer_text(result: dict[str, Any]) -> str:
    for key in ("final_answer", "summary", "answer", "output"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = list(result.get("messages") or [])
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content") or "").strip()
    return str(getattr(last, "content", "") or "").strip()


def _citation_has_provenance(citation: dict[str, Any]) -> bool:
    source = citation.get("source") or citation.get("title")
    locator = citation.get("url") or citation.get("snippet") or citation.get(
        "evidence_id"
    )
    return bool(source and locator)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    expected: Any = None,
    actual: Any = None,
) -> None:
    item = {"name": name, "passed": passed}
    if expected is not None:
        item["expected"] = expected
    if actual is not None:
        item["actual"] = actual
    checks.append(item)


__all__ = ["evaluate_run"]
