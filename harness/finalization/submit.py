"""把助手正文写成用户可见回答。不做 evidence 绑定闸门。"""

from __future__ import annotations

import re
from typing import Sequence

from compliance.review import review_answer
from harness.session.types import SessionEvent

COMPLIANCE_FALLBACK = (
    "该回答因合规原因无法展示。请换个问法，不要询问具体买卖指令或承诺收益。"
)


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
    markdown = str(text or "").strip()
    if not markdown:
        return None
    decision = review_answer(markdown)
    if getattr(decision, "action", "pass") == "block":
        return COMPLIANCE_FALLBACK
    return markdown
