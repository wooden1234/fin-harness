"""基于已收集 Evidence 成稿：优先 finalign，不可用则 DeepSeek 成稿2。"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from app.core.config import settings
from app.core.logger import get_logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_faq_llm, get_finance_llm, is_finance_llm_available
from agents.main_deep_agent.contracts import MainAgentResponse
from agents.main_deep_agent.state import MainAgentProgressJournal

logger = get_logger(service="main_deep_agent_drafting")

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)
_EVIDENCE_ID_RE = re.compile(r"\bmain:[A-Za-z0-9._:-]+")
_MAX_EVIDENCE_ITEMS = 24
_MAX_FACT_CHARS = 240
_MAX_NARRATIVE_CHARS = 600
_MAX_MATERIALS_CHARS = 12_000

DraftEngine = Literal["finalign", "deepseek"]

_DRAFT_SYSTEM = """你是金融研究成稿助手。只依据用户问题、材料与草稿修订答案。
规则：
1. 不得编造材料未出现的数值、公司、来源、驱动因素或结论。
2. 对照用户问题逐项作答：材料已覆盖的写 statements/tables；材料未覆盖的子问题必须写入 gaps，明确写「未披露/未检索到…」，禁止用空泛表述或相邻指标冒充。
3. 输出必须是单个 JSON 对象，字段仅限：
   mode, direct_answer, clarification, heading, statements, tables, gaps, follow_ups
4. mode 只能是 direct / grounded / clarify。
5. grounded 时优先用 statements/tables，并保留材料中的 evidence_id；表格只放问题所需且材料已给出的指标，不要为凑完整擅自加行。
5b. statements 每项必须是 {"text":"...","statement_type":"fact","evidence_ids":["main:..."]}；禁止把 statement 写成字符串，禁止使用单数 evidence_id。
5a. 材料里的 narrative 字段是原始公告/工具返回的定性说明文字（如变动原因、driver 描述），驱动因素、不确定性等定性结论必须优先从 narrative 提炼，不得因为它不在 facts 数值里就当作「材料未覆盖」写入 gaps 或凭空编造。
6. 注意数学计算：表内金额、同比/增幅上下限与中枢必须同源自洽；展示值若经四舍五入，变动比例须按同一精度重算或直接引用材料原比例，禁止出现「用表上展示数反算对不上」的口径。无法核对时写入 gaps，不要硬凑。
7. follow_ups 最多 3 条，且只能追问材料已涉及、正文已陈述的主题；材料没有的主题（如毛利率、海外占比）一律不要生成追问。
8. 不要输出 JSON 以外的文字。"""


def _compact_evidence_materials(journal: MainAgentProgressJournal) -> str:
    lines: list[str] = []
    for item in list(journal.evidence.values())[:_MAX_EVIDENCE_ITEMS]:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        display = str(metadata.get("display_text") or item.title or "").strip()
        display = " ".join(display.split())[:800]
        fact_bits: list[str] = []
        for fact in list(metadata.get("facts") or [])[:8]:
            if not isinstance(fact, dict):
                continue
            metric = str(fact.get("metric") or "").strip()
            value = fact.get("value")
            unit = str(fact.get("unit") or "").strip()
            period = str(fact.get("period") or fact.get("fiscal_period") or "").strip()
            piece = " ".join(
                part for part in (metric, str(value) if value is not None else "", unit, period) if part
            ).strip()
            if piece:
                fact_bits.append(piece[:_MAX_FACT_CHARS])
        narrative = str(metadata.get("narrative") or "").strip()
        narrative = " ".join(narrative.split())[:_MAX_NARRATIVE_CHARS]
        chunk = {
            "evidence_id": item.evidence_id,
            "title": item.title or "",
            "provider": item.provider or "",
            "display_text": display,
            "facts": fact_bits,
        }
        if narrative:
            chunk["narrative"] = narrative
        lines.append(json.dumps(chunk, ensure_ascii=False))
    text = "\n".join(lines)
    if len(text) > _MAX_MATERIALS_CHARS:
        return text[:_MAX_MATERIALS_CHARS] + "\n…(材料已截断)"
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _text_value(value: object) -> str:
    """从常见模型包装对象中提取文本，不序列化未知复杂结构。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = [
            item.strip()
            for item in value.values()
            if isinstance(item, str) and item.strip()
        ]
        return "；".join(parts)
    return ""


def _evidence_ids(value: object, *, text: str = "") -> list[str]:
    """兼容单数/复数 Evidence 字段，并从陈述文本中提取显式 ID。"""
    raw_items = value if isinstance(value, list) else [value]
    ids = [
        str(item).strip()
        for item in raw_items
        if isinstance(item, str) and str(item).strip()
    ]
    ids.extend(_EVIDENCE_ID_RE.findall(text))
    return list(dict.fromkeys(ids))[:8]


def _normalize_statement(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, str):
        text = raw.strip()
        return {"text": text, "evidence_ids": _evidence_ids(None, text=text)} if text else None
    if not isinstance(raw, dict):
        return None
    text = next(
        (
            _text_value(raw.get(key))
            for key in ("text", "content", "claim", "statement", "结论", "说明")
            if _text_value(raw.get(key))
        ),
        "",
    )
    if not text:
        metadata_keys = {
            "evidence_id", "evidence_ids", "statement_type", "type",
            "entity", "metric", "period", "unit",
        }
        text_candidates = [
            value.strip()
            for key, value in raw.items()
            if key not in metadata_keys
            and isinstance(value, str)
            and value.strip()
            and not value.strip().startswith("main:")
        ]
        text = max(text_candidates, key=len, default="")
    if not text:
        return None
    statement_type = str(raw.get("statement_type") or raw.get("type") or "fact")
    if statement_type not in {"fact", "calculation", "inference", "caveat"}:
        statement_type = "fact"
    return {
        "text": text,
        "statement_type": statement_type,
        "evidence_ids": _evidence_ids(
            raw.get("evidence_ids", raw.get("evidence_id")),
            text=text,
        ),
        "entity": _text_value(raw.get("entity")),
        "metric": _text_value(raw.get("metric")),
        "period": _text_value(raw.get("period")),
        "unit": _text_value(raw.get("unit")),
    }


def _normalize_table(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    columns = [str(item).strip() for item in list(raw.get("columns") or []) if str(item).strip()]
    if len(columns) < 2:
        return None
    table_evidence_ids = _evidence_ids(raw.get("evidence_ids", raw.get("evidence_id")))
    rows: list[dict[str, Any]] = []
    for raw_row in list(raw.get("rows") or []):
        if isinstance(raw_row, list):
            cells = [str(item).strip() for item in raw_row]
            row_evidence_ids = table_evidence_ids
        elif isinstance(raw_row, dict):
            cells = [str(item).strip() for item in list(raw_row.get("cells") or [])]
            row_evidence_ids = _evidence_ids(
                raw_row.get("evidence_ids", raw_row.get("evidence_id"))
            ) or table_evidence_ids
        else:
            continue
        if len(cells) == len(columns) and all(cells):
            rows.append({"cells": cells, "evidence_ids": row_evidence_ids})
    if not rows:
        return None
    return {
        "title": _text_value(raw.get("title")),
        "columns": columns,
        "rows": rows,
    }


def _string_list(value: object, *, limit: int) -> list[str]:
    raw_items = value if isinstance(value, list) else [value]
    return [
        text
        for item in raw_items
        if (text := _text_value(item))
    ][:limit]


def _normalize_draft_payload(data: dict[str, Any]) -> dict[str, Any]:
    """把模型常见近似结构收敛到严格 MainAgentResponse 契约。"""
    statements = [
        item
        for raw in list(data.get("statements") or [])
        if (item := _normalize_statement(raw)) is not None
    ]
    tables = [
        item
        for raw in list(data.get("tables") or [])
        if (item := _normalize_table(raw)) is not None
    ]
    direct_answer = _text_value(data.get("direct_answer"))
    mode = str(data.get("mode") or "").strip()
    if mode not in {"direct", "grounded", "clarify"}:
        mode = "grounded" if statements or tables else "direct"
    if mode == "grounded":
        direct_answer = ""
    return {
        "mode": mode,
        "direct_answer": direct_answer,
        "clarification": _text_value(data.get("clarification")),
        "heading": _text_value(data.get("heading")),
        "statements": statements,
        "tables": tables,
        "gaps": _string_list(data.get("gaps"), limit=20),
        "follow_ups": _string_list(data.get("follow_ups"), limit=3),
    }


def _build_user_prompt(
    *,
    query: str,
    materials: str,
    response: MainAgentResponse | None,
) -> str:
    draft_payload = (
        response.model_dump(mode="json")
        if response is not None
        else {"mode": "grounded", "statements": [], "tables": [], "gaps": ["材料有限"]}
    )
    question = (query or "").strip() or "（未提供原问题，请严格按材料保守成稿）"
    return (
        "请仅依据材料回答用户问题；可参考草稿，但不得引入材料外事实。\n"
        "先拆解问题要点，再逐点对照材料：有证据则写结论，无证据则写入 gaps。\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<materials>\n{materials or '（无结构化证据，仅可基于草稿做保守表述）'}\n</materials>\n\n"
        f"<draft>\n{json.dumps(draft_payload, ensure_ascii=False)}\n</draft>"
    )


async def _draft_with_llm(
    *,
    llm: BaseChatModel,
    engine: DraftEngine,
    query: str,
    materials: str,
    response: MainAgentResponse | None,
) -> MainAgentResponse | None:
    """单次成稿调用；失败返回 None。"""
    try:
        result = await llm.ainvoke(
            [
                SystemMessage(content=_DRAFT_SYSTEM),
                HumanMessage(
                    content=_build_user_prompt(
                        query=query,
                        materials=materials,
                        response=response,
                    )
                ),
            ]
        )
        content = result.content if isinstance(result.content, str) else str(result.content)
        data = _extract_json_object(content)
        if data is None:
            logger.warning("{} draft returned non-json", engine)
            return None
        refined = MainAgentResponse.model_validate(_normalize_draft_payload(data))
        logger.info(
            "{} draft applied: mode={} statements={} tables={} gaps={}",
            engine,
            refined.mode,
            len(refined.statements),
            len(refined.tables),
            len(refined.gaps),
        )
        return refined
    except Exception as exc:
        detail = str(exc)[:400]
        logger.warning("{} draft failed: {} detail={}", engine, type(exc).__name__, detail)
        return None


async def refine_main_response_with_finance_llm(
    response: MainAgentResponse | None,
    journal: MainAgentProgressJournal,
    *,
    query: str = "",
) -> MainAgentResponse | None:
    """工具结果之后成稿：优先 finalign；不可用/失败则 DeepSeek 成稿2。"""
    if not bool(settings.FINANCE_LLM_DRAFT_ENABLED):
        return response

    materials = _compact_evidence_materials(journal)
    if not materials and response is None:
        return response

    if is_finance_llm_available():
        refined = await _draft_with_llm(
            llm=get_finance_llm(),
            engine="finalign",
            query=query,
            materials=materials,
            response=response,
        )
        if refined is not None:
            return refined
        logger.warning("finalign draft unavailable/failed, fallback to deepseek draft")
    else:
        logger.info("finalign unavailable, use deepseek draft (成稿2)")

    refined = await _draft_with_llm(
        llm=get_faq_llm(),
        engine="deepseek",
        query=query,
        materials=materials,
        response=response,
    )
    return refined if refined is not None else response


__all__ = ["refine_main_response_with_finance_llm"]
