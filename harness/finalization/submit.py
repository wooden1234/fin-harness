"""把助手正文写成用户可见回答。不做 evidence 绑定闸门。"""

from __future__ import annotations

import re
from typing import Sequence

from compliance.review import review_answer
from harness.session.types import SessionEvent

COMPLIANCE_FALLBACK = (
    "该回答因合规原因无法展示。请换个问法，不要询问具体买卖指令或承诺收益。"
)

_EVIDENCE_PAREN_RE = re.compile(
    r"\s*[（(]\s*evidence_id\s*[:：][^）)\n]*[）)]",
    re.IGNORECASE,
)
_SOURCE_CLAUSE_RE = re.compile(
    r"[ \t]*(?:\*\*|__)?数据来源\s*[:：][^\n]*"
)


def strip_source_attribution(text: str) -> str:
    """去掉正文里的数据来源说明和 evidence_id，核对信息留在工具结果中。"""
    cleaned = _EVIDENCE_PAREN_RE.sub("", text)
    cleaned = _SOURCE_CLAUSE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def collect_evidence(events: Sequence[SessionEvent], *, turn: int) -> set[str]:
    """从本轮 tool/result 收集 evidence_id，供日志与评测使用。"""
    found: set[str] = set()
    for event in events:
        if event.turn != turn or event.event_type != "tool/result":
            continue
        evidence_id = event.data.get("evidence_id")
        if evidence_id:
            found.add(str(evidence_id))
        content = str(event.data.get("content") or "")
        for match in re.findall(r'"evidence_id"\s*:\s*"([^"]+)"', content):
            found.add(match)
    return found


def finalize_markdown(text: str) -> str | None:
    """空正文不发布；合规命中则换成安全提示。"""
    markdown = strip_source_attribution(str(text or ""))
    if not markdown:
        return None
    decision = review_answer(markdown)
    if getattr(decision, "action", "pass") == "block":
        return COMPLIANCE_FALLBACK
    return markdown
