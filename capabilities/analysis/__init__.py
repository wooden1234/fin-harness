"""用本地 finalign 对多工具结果做综合分析。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from capabilities.evidence import stamp_evidence

TOOL_ID = "finalign.analyze"
TOOL_NAME = "finalign_analyze"
MAX_MATERIALS = 12
MAX_MATERIAL_CHARS = 6000


def _error(code: str, **extra: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": code}
    payload.update(extra)
    return payload


ANALYSIS_SYSTEM = (
    "你是面向投资者的金融分析助手，只根据给定材料作答。"
    "数字、区间、同比必须与材料一致，禁止用常识补全或改数量级。"
    "多源冲突时点明差异与口径，不要抹平。"
    "材料不足就说暂未查到，不要编造。"
    "不要输出检索过程、工具名或 evidence_id；用简洁中文直接回答用户问题。"
    "不得给出个性化买卖指令，不得承诺收益。"
)


def _truncate(text: str, limit: int = MAX_MATERIAL_CHARS) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 16] + "\n…[truncated]"


def normalize_materials(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    items: list[dict[str, Any]] = []
    for entry in raw[:MAX_MATERIALS]:
        if isinstance(entry, str):
            content = _truncate(entry)
            if content:
                items.append({"tool": "", "ok": True, "evidence_id": None, "content": content})
            continue
        if not isinstance(entry, Mapping):
            continue
        content = _truncate(str(entry.get("content") or entry.get("text") or ""))
        if not content:
            continue
        evidence_id = entry.get("evidence_id")
        items.append(
            {
                "tool": str(entry.get("tool") or entry.get("name") or "").strip(),
                "ok": bool(entry.get("ok", True)),
                "evidence_id": str(evidence_id) if evidence_id else None,
                "content": content,
            }
        )
    return items


def _format_materials(materials: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(materials, start=1):
        tool = str(item.get("tool") or "tool").strip() or "tool"
        evidence_id = str(item.get("evidence_id") or "").strip()
        status = "ok" if item.get("ok", True) else "failed"
        header = f"### 材料 {index} · {tool} · {status}"
        if evidence_id:
            header += f" · evidence_id={evidence_id}"
        blocks.append(f"{header}\n{item.get('content') or ''}")
    return "\n\n".join(blocks)


def _message_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(str(item.get("text") or ""))
        return "".join(parts).strip()
    return str(content or "").strip()


async def synthesize_answer(
    *,
    question: str,
    materials: Sequence[Mapping[str, Any]] | Sequence[str] | None = None,
) -> dict[str, Any]:
    query = str(question or "").strip()
    if not query:
        return _error("empty_question")
    normalized = normalize_materials(list(materials or []))
    if not normalized:
        return _error("no_evidence_to_analyze")

    from app.core.config import settings

    if not bool(settings.FINANCE_LLM_DRAFT_ENABLED):
        return _error(
            "draft_disabled",
            message="分析模型未启用，请根据已检索材料直接用正文回答用户。",
        )

    from agents.llm import get_finance_llm, is_finance_llm_available

    if not is_finance_llm_available():
        return _error(
            "finalign_unavailable",
            message=(
                "本地 finalign 未启动，分析工具不回退 DeepSeek，以免另开一条冷前缀打乱规划 KV。"
                "请根据本轮已检索的工具结果直接用正文回答用户，不要再调用 finalign_analyze。"
            ),
        )

    model_name = str(settings.FINANCE_LLM_MODEL or "finalign")
    human = (
        f"用户问题：\n{query}\n\n"
        "请只根据下列工具结果给出最终回答。\n\n"
        f"{_format_materials(normalized)}"
    )
    try:
        llm = get_finance_llm()
        result = await llm.ainvoke(
            [
                ("system", ANALYSIS_SYSTEM),
                ("human", human),
            ]
        )
        markdown = _message_text(result)
    except Exception as exc:  # noqa: BLE001
        return _error(
            "analysis_failed",
            message=str(exc)[:400],
        )
    if not markdown:
        return _error("empty_analysis")

    evidence_ids = [str(item["evidence_id"]) for item in normalized if item.get("evidence_id")]
    payload = {
        "ok": True,
        "content": markdown,
        "markdown": markdown,
        "question": query,
        "evidence_ids": evidence_ids,
        "model": model_name,
        "used_finalign": True,
        "display_text": markdown[:800],
    }
    return stamp_evidence(TOOL_ID, payload)
