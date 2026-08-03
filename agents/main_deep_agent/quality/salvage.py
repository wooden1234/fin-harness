"""结构化答案失败时的 Evidence 降级渲染。"""

from __future__ import annotations

from collections import defaultdict

from agents.main_deep_agent.quality.renderer import _clean_display_text
from agents.orchestrator.contracts import Evidence
from agents.orchestrator.state import OrchestratorState

def _render_salvage(
    state: OrchestratorState,
) -> tuple[str, list[str]]:
    """按实体渲染清洗后的短要点，绝不回显内部 JSON 或通稿长文。"""
    evidence = list(state.get("evidence") or [])
    displayable = [(item, _clean_display_text(item)) for item in evidence]
    displayable = [(item, text) for item, text in displayable if text]
    # 同实体优先官方/结构化，并限制条数，避免刷屏
    ranked: list[tuple[Evidence, str]] = []
    seen_text: set[str] = set()
    for item, text in sorted(
        displayable,
        key=lambda pair: (
            0 if not pair[0].metadata.get("is_preview") else 1,
            0 if pair[0].metadata.get("source_grade") in {"official", "structured"} else 1,
            0 if pair[0].metadata.get("facts") else 1,
        ),
    ):
        key = text[:120]
        if key in seen_text:
            continue
        seen_text.add(key)
        ranked.append((item, text))
    gaps = ["模型未能生成可验证的结构化答案，已仅保留工具返回的可展示资料。"]
    if bool(state.get("investment_action_sensitive")):
        gaps.append("以上仅梳理公开信息中的风险与观察点，不构成买卖建议。")
    if not ranked:
        return (
            "这轮还没拿到可展示、可核对的资料，暂时无法形成可靠答案。\n\n"
            "### 还缺什么\n" + "\n".join(f"- {message}" for message in gaps),
            gaps,
        )

    grouped: dict[str, list[tuple[Evidence, str]]] = defaultdict(list)
    for item, text in ranked[:8]:
        entity = str(item.metadata.get("entity") or "已核验资料")
        if len(grouped[entity]) >= 3:
            continue
        grouped[entity].append((item, text))
    lines = [
        "本轮未能生成结构化答案，先列出已核对到的要点：",
        "",
        "### 已核验资料",
    ]
    citation_index = 0
    for entity, items in grouped.items():
        if entity != "已核验资料":
            lines.append(f"#### {entity}")
        for _, text in items:
            citation_index += 1
            lines.append(f"- {text}[{citation_index}]")
    lines.append("### 还缺什么")
    lines.extend(f"- {message}" for message in gaps)
    return "\n".join(lines), gaps

render_salvage = _render_salvage

__all__ = ["render_salvage"]
