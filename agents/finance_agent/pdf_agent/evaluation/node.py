"""PDF 检索证据评判节点。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm import get_pdf_llm
from agents.finance_agent.pdf_agent.generation import extract_citation_indices

from ..state import PdfAgentState
from ..trace import append_trace
from .prompt import PDF_EVIDENCE_EVALUATION_PROMPT

_ROUTES = {"answer", "rewrite", "web_search"}
_STRATEGIES = {"none", "step_back", "hyde", "answer_mismatch"}


def _parse_evaluation(content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else str(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    if not isinstance(payload, dict):
        return {}

    route = str(payload.get("route") or "web_search").strip().lower()
    if route not in _ROUTES:
        route = "web_search"
    strategy = str(payload.get("next_strategy") or "none").strip().lower()
    if strategy not in _STRATEGIES:
        strategy = "step_back" if route == "rewrite" else "none"
    try:
        confidence = min(max(float(payload.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "route": route,
        "relevance": payload.get("relevance") is True,
        "completeness": payload.get("completeness") is True,
        "ambiguity": payload.get("ambiguity") is True,
        "answerable": payload.get("answerable") is True,
        "next_strategy": strategy,
        "reason": str(payload.get("reason") or ""),
        "missing_fields": _string_list(payload.get("missing_fields")),
        "unsupported_facts": _string_list(payload.get("unsupported_facts")),
        "strategy_reason": str(payload.get("strategy_reason") or ""),
        "web_reason": str(payload.get("web_reason") or ""),
        "confidence": confidence,
        "answer": str(payload.get("answer") or "").strip(),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


async def evaluate_evidence_node(state: PdfAgentState, *, config=None) -> PdfAgentState:
    question = str(state.get("original_query") or state.get("query") or "").strip()
    context = str(state.get("context") or "").strip()
    prompt = PDF_EVIDENCE_EVALUATION_PROMPT.format(question=question, context=context)
    try:
        response = await get_pdf_llm().ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="请返回证据判断与答案的 JSON 结果。"),
            ],
            config=config,
        )
        evaluation = _parse_evaluation(response.content)
        if not evaluation:
            raise ValueError("证据评判返回无法解析")
        answer = evaluation["answer"] if evaluation["route"] == "answer" else ""
        if evaluation["route"] == "answer" and not answer:
            evaluation = {
                **evaluation,
                "route": "web_search",
                "answerable": False,
                "reason": evaluation["reason"] or "模型未生成有效答案",
                "web_reason": evaluation["web_reason"] or "PDF 证据无法形成有效答案",
            }
        citation_indices = extract_citation_indices(answer, len(state.get("hits") or []))
        trace_update = append_trace(
            state,
            "evidence_evaluate",
            status="ok",
            route=evaluation["route"],
            relevance=evaluation["relevance"],
            completeness=evaluation["completeness"],
            ambiguity=evaluation["ambiguity"],
            answerable=evaluation["answerable"],
            missing_fields=evaluation["missing_fields"],
            unsupported_facts=evaluation["unsupported_facts"],
            strategy_reason=evaluation["strategy_reason"],
            web_reason=evaluation["web_reason"],
            confidence=evaluation["confidence"],
            answer_chars=len(answer),
            citation_count=len(citation_indices),
        )
        return {
            "evidence_evaluation": evaluation,
            "evidence_evaluation_status": "ok",
            "evidence_route": evaluation["route"],
            "next_rewrite_strategy": evaluation["next_strategy"],
            "answer": answer,
            "citation_indices": citation_indices,
            **trace_update,
        }
    except Exception as exc:
        trace_update = append_trace(state, "evidence_evaluate", status="unavailable", error=str(exc))
        return {
            "evidence_evaluation": {"route": "web_search", "reason": str(exc)},
            "evidence_evaluation_status": "unavailable",
            "evidence_route": "web_search",
            "next_rewrite_strategy": "none",
            **trace_update,
        }
