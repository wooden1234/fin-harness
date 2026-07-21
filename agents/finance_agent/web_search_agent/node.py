"""Web Search Agent 节点：受控联网检索兜底。"""

from __future__ import annotations

from typing import Any, NotRequired
from typing_extensions import TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.finance_agent.web_search_agent.prompts import (
    WEB_SEARCH_BUSY_ANSWER,
    WEB_SEARCH_NO_CONFIG_ANSWER,
    WEB_SEARCH_NO_RESULT_ANSWER,
)
from agents.states import Citation, FinAgentState
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="web_search_agent")


class WebSearchResult(TypedDict):
    title: NotRequired[str]
    url: NotRequired[str]
    content: NotRequired[str]
    published_date: NotRequired[str]
    score: NotRequired[float]


class WebSearchResponse(TypedDict):
    answer: str
    results: list[WebSearchResult]
    configured: bool


def _latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _query_from_state(state: FinAgentState) -> str:
    sub_question = str(state.get("sub_question") or "")
    if sub_question:
        return sub_question

    sub_task_id = str(state.get("sub_task_id") or "")
    for result in reversed(list(state.get("task_results") or [])):
        if sub_task_id and result.get("sub_task_id") != sub_task_id:
            continue
        question = str(result.get("question") or "")
        if question:
            return question

    return _latest_user_query(list(state.get("messages") or []))


def _to_citations(results: list[WebSearchResult], *, sub_task_id: str = "") -> list[Citation]:
    citations: list[Citation] = []
    for result in results:
        url = str(result.get("url") or "")
        title = str(result.get("title") or url or "联网搜索结果")
        snippet = str(result.get("content") or "")[:300]
        citation: Citation = {
            "source": title,
            "title": title,
            "url": url,
            "snippet": snippet,
            "source_type": "web",
            "sub_task_id": sub_task_id,
        }
        published_at = result.get("published_date")
        if published_at:
            citation["published_at"] = str(published_at)
        citations.append(citation)
    return citations


def _format_sources(citations: list[Citation]) -> str:
    """供内部 context 使用的来源摘要：只用标题，不拼原始 URL（避免 /goto? 脏链进回答）。"""
    lines = []
    for index, citation in enumerate(citations, start=1):
        title = citation.get("title") or citation.get("source") or "来源"
        lines.append(f"[{index}] {title}")
    return "\n".join(lines)


def _fallback_answer_from_results(results: list[WebSearchResult]) -> str:
    parts = []
    for index, result in enumerate(results, start=1):
        title = str(result.get("title") or f"搜索结果 {index}")
        content = str(result.get("content") or "").strip()
        if content:
            parts.append(f"[{index}] {title}：{content}")
    return "\n".join(parts)


async def _search_tavily(query: str) -> WebSearchResponse:
    if not settings.TAVILY_API_KEY:
        return {
            "answer": WEB_SEARCH_NO_CONFIG_ANSWER,
            "results": [],
            "configured": False,
        }

    max_results = max(1, min(settings.WEB_SEARCH_MAX_RESULTS, 10))
    payload: dict[str, Any] = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(settings.TAVILY_SEARCH_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    results = [
        WebSearchResult(
            title=str(item.get("title") or ""),
            url=str(item.get("url") or ""),
            content=str(item.get("content") or ""),
            published_date=str(item.get("published_date") or ""),
            score=float(item.get("score") or 0),
        )
        for item in list(data.get("results") or [])
        if isinstance(item, dict)
    ]
    answer = str(data.get("answer") or "").strip()
    return {"answer": answer, "results": results, "configured": True}


async def search_web(query: str) -> WebSearchResponse:
    provider = settings.WEB_SEARCH_PROVIDER.lower()
    if provider != "tavily":
        logger.warning("unsupported web search provider={}", settings.WEB_SEARCH_PROVIDER)
        return {
            "answer": WEB_SEARCH_NO_CONFIG_ANSWER,
            "results": [],
            "configured": False,
        }
    return await _search_tavily(query)


def _is_chain_fallback(state: FinAgentState) -> bool:
    """证据链降级到 web 时，不向 messages 推送中间答案，交给 summarize。"""
    chain = [str(item) for item in (state.get("evidence_chain") or []) if str(item)]
    return bool(chain) and chain[0] != "web_search"


def _public_messages(answer: str, *, suppress: bool) -> list:
    return [] if suppress else [AIMessage(content=answer)]


async def web_search_agent(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    query = _query_from_state(state)
    sub_task_id = str(state.get("sub_task_id") or "")
    suppress_messages = _is_chain_fallback(state)

    logger.info("web_search query={} sub_task_id={}", query[:80], sub_task_id)

    if not query:
        answer = WEB_SEARCH_NO_RESULT_ANSWER
        return {
            "messages": [],
            "citations": [],
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "web_search",
                    "context": answer,
                    "coverage": "uncovered",
                    "fallback_reason": "web_empty_query",
                }
            ],
            "steps": ["web_search_agent"],
        }

    try:
        search_response = await search_web(query)
    except Exception:
        logger.exception("web search failed")
        answer = WEB_SEARCH_BUSY_ANSWER
        return {
            "messages": [],
            "citations": [],
            "task_results": [
                {
                    "sub_task_id": sub_task_id,
                    "question": query,
                    "type": "web_search",
                    "context": answer,
                    "coverage": "uncovered",
                    "fallback_reason": "web_search_failed",
                }
            ],
            "steps": ["web_search_agent"],
        }

    citations = _to_citations(search_response["results"], sub_task_id=sub_task_id)
    if search_response["answer"]:
        answer = search_response["answer"]
    elif citations:
        answer = _fallback_answer_from_results(search_response["results"])
    else:
        answer = search_response["answer"] or WEB_SEARCH_NO_RESULT_ANSWER

    # 无搜索结果（未配置 / 无可靠来源）时明确 uncovered，交给 summarize 说明依据不足
    coverage = "covered" if citations else "uncovered"

    # 引用交给 citations 字段给前端展示，不把 URL 列表塞进回答正文
    context = f"[联网搜索] {answer}"
    if citations:
        context = f"{context}\n相关来源标题：\n{_format_sources(citations)}"

    return {
        "messages": _public_messages(
            answer,
            suppress=suppress_messages or coverage == "uncovered",
        ),
        "citations": citations,
        "task_results": [
            {
                "sub_task_id": sub_task_id,
                "question": query,
                "type": "web_search",
                "context": context,
                "citations": citations,
                "coverage": coverage,
                **({} if citations else {"fallback_reason": "web_no_result"}),
            }
        ],
        "steps": ["web_search_agent"],
    }
