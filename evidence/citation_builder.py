"""引用归一化。"""

from __future__ import annotations

from typing import Any

from evidence.models import Evidence


def build_evidence_from_citations(citations: list[dict[str, Any]]) -> list[Evidence]:
    result: list[Evidence] = []
    for item in citations:
        result.append(
            Evidence(
                source=str(item.get("source") or ""),
                snippet=str(item.get("snippet") or ""),
                url=str(item.get("url") or ""),
                page=item.get("page"),
                source_type=str(item.get("source_type") or ""),
                metadata=dict(item),
            )
        )
    return result
