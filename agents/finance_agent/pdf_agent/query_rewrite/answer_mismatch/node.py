"""答案与问题不匹配时的查询改写节点。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_pdf_llm

from ...state import PdfAgentState
from ...trace import append_trace
from .prompt import PDF_ANSWER_MISMATCH_PROMPT


_EXCLUSION_RE = re.compile(
    r"(?:[，,。；;]\s*)?(?:不要答成|不要使用|不要用|不是|而非)\s*([^，,。；;？?]+)"
)
_QUESTION_FILLERS_RE = re.compile(
    r"(?:请问|请查询|帮我查询|帮我查|告诉我|分别是多少|是多少|是否|如何|怎么样)"
)
_PUNCT_RE = re.compile(r"[\s，,。；;：:？?！!、]+")


@dataclass(frozen=True, slots=True)
class RewriteContract:
    base_query: str
    must_not: tuple[str, ...]
    anchor: str


def _compact(text: str) -> str:
    return _PUNCT_RE.sub("", str(text or "")).strip()


def forbidden_variants(value: str) -> tuple[str, ...]:
    """保留完整排除短语，同时抽取其中的错误主体/年份，防止同义改写绕过。"""
    compact = _compact(value)
    if not compact:
        return ()
    variants = [compact]
    stem = re.sub(
        r"(?:营业收入|收入|成本|年度报告|年报|总指数|GDP.*|CPI.*|PPI.*|LPR.*)$",
        "",
        compact,
        flags=re.IGNORECASE,
    ).strip()
    if len(stem) >= 2 and stem not in variants:
        variants.append(stem)
    return tuple(variants)


def _clean_base_query(question: str) -> str:
    without_exclusions = _EXCLUSION_RE.sub("", str(question or ""))
    without_fillers = _QUESTION_FILLERS_RE.sub("", without_exclusions)
    return re.sub(r"\s+", " ", without_fillers).strip(" ，,。；;：:？?！!")


def _select_anchor(base_query: str) -> str:
    text = str(base_query or "")
    if "指数" in text:
        return "分指数表"
    if any(term in text for term in ("成本", "占比", "毛利率")):
        return "成本构成"
    if (
        any(term in text.upper() for term in ("GDP", "CPI", "PPI", "LPR"))
        or "经济" in text
    ):
        return "主要经济指标"
    if any(
        term in text
        for term in (
            "营业收入",
            "营收",
            "净利润",
            "研发投入",
            "研发费用",
            "现金流",
            "资产",
            "负债",
        )
    ):
        return "年度报告 主要财务指标"
    if any(term in text for term in ("出货", "产品进展")):
        return "产品进展"
    return ""


def build_rewrite_contract(question: str) -> RewriteContract:
    must_not = tuple(
        value.strip()
        for value in _EXCLUSION_RE.findall(str(question or ""))
        if value.strip()
    )
    base_query = _clean_base_query(question) or str(question or "").strip()
    return RewriteContract(
        base_query=base_query,
        must_not=must_not,
        anchor=_select_anchor(base_query),
    )


def _fallback_query(contract: RewriteContract) -> str:
    return " ".join(item for item in (contract.base_query, contract.anchor) if item).strip()


def _parse_model_query(content: object) -> tuple[str, float | None]:
    raw = content if isinstance(content, str) else str(content or "")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    payload: object | None = None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except ValueError:
                payload = None
    if isinstance(payload, dict):
        query = str(payload.get("query") or "").strip()
        try:
            confidence = float(payload["confidence"]) if "confidence" in payload else None
        except (TypeError, ValueError):
            confidence = None
        return query, confidence
    # 兼容尚未切换到 JSON 输出的模型，仍需通过后续确定性校验。
    return raw, None


def validate_rewrite_query(candidate: str, contract: RewriteContract) -> tuple[bool, str]:
    compact_candidate = _compact(candidate)
    if not compact_candidate:
        return False, "empty"
    if _compact(contract.base_query) not in compact_candidate:
        return False, "missing_required_query"
    for forbidden in contract.must_not:
        if any(variant in compact_candidate for variant in forbidden_variants(forbidden)):
            return False, "forbidden_term"
    if contract.anchor and not any(
        _compact(part) in compact_candidate for part in contract.anchor.split() if _compact(part)
    ):
        return False, "missing_anchor"
    if len(candidate) > max(96, len(contract.base_query) + 48):
        return False, "too_long"
    return True, "ok"


async def answer_mismatch_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    question = str(state.get("original_query") or state.get("query") or "").strip()
    context = str(state.get("context") or "").strip()
    contract = build_rewrite_contract(question)
    prompt = PDF_ANSWER_MISMATCH_PROMPT.format(
        question=question,
        context=context,
        must_keep=contract.base_query,
        must_not="、".join(contract.must_not) or "无",
        anchor=contract.anchor or "无；不要强行添加",
    )
    model_rewrite = ""
    confidence: float | None = None
    validation_reason = "llm_error"
    try:
        response = await get_pdf_llm().ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=question)],
            config=config,
        )
        model_rewrite, confidence = _parse_model_query(response.content)
        valid, validation_reason = validate_rewrite_query(model_rewrite, contract)
    except Exception:
        valid = False

    rewritten = model_rewrite if valid else _fallback_query(contract)
    used_fallback = not valid

    trace_update = append_trace(
        state,
        "answer_mismatch",
        status="validated_fallback" if used_fallback else "ok",
        rewrite_query=rewritten,
        model_rewrite=model_rewrite,
        validation_reason=validation_reason,
        must_keep=[contract.base_query],
        must_not=list(contract.must_not),
        anchor=contract.anchor,
        confidence=confidence,
        rewrite_count=int(state.get("rewrite_count") or 0) + 1,
    )
    return {
        "query": rewritten,
        "rewrite_query": rewritten,
        "rewrite_strategy": "answer_mismatch",
        "rewrite_reason": (
            "question_context_mismatch_fallback" if used_fallback else "question_context_mismatch"
        ),
        "rewrite_count": int(state.get("rewrite_count") or 0) + 1,
        **trace_update,
    }
