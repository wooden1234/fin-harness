"""Analyzer 前置的三态问题补全：直通、改写或澄清。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
from typing import Literal, Mapping
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.context import project_conversation_context
from agents.context_compressor.structured import parse_summary_v2
from agents.llm import get_router_llm
from agents.query_rewrite.prompts import REWRITE_HUMAN_PROMPT, REWRITE_SYSTEM_PROMPT
from agents.states import FinAgentState
from app.core.logger import get_logger

logger = get_logger(service="query_rewrite")

_RECENT_MESSAGE_LIMIT = 8
RewriteAction = Literal["passthrough", "rewrite", "uncertain"]
RewriteReasonCode = Literal[
    "empty_query",
    "self_contained",
    "context_reference",
    "followup_prefix",
    "short_followup",
    "context_missing",
    "context_irrelevant",
    "multiple_entity_candidates",
    "clarification_reply",
    "clarification_cancelled",
    "clarification_expired",
    "model_uncertain",
    "model_error",
    "rewrite_validation_failed",
]
_CONTEXT_REFERENCE_MARKERS = (
    "它",
    "该公司",
    "这家公司",
    "上述",
    "上面",
    "前者",
    "后者",
    "其中",
    "这些",
    "那些",
    "这个",
    "那个",
    "刚才",
    "之前",
    "上一步",
)
_FOLLOWUP_PREFIXES = (
    "那",
    "再",
    "继续",
    "另外",
    "然后",
    "顺便",
    "改成",
    "改为",
    "换成",
    "只看",
    "再看",
    "再分析",
)
_SHORT_FOLLOWUP_SUFFIXES = (
    "呢",
    "如何",
    "怎么样",
    "多少",
    "为什么",
    "怎么看",
    "有什么风险",
)
_MAX_SHORT_FOLLOWUP_LENGTH = 18
_SINGULAR_AMBIGUOUS_MARKERS = (
    "它",
    "该公司",
    "这家公司",
    "这个",
    "那个",
)
_ALWAYS_DEPENDENT_PREFIXES = (
    "继续",
    "然后",
    "改成",
    "改为",
    "换成",
    "只看",
    "再看",
    "再分析",
)
_DISCOURSE_PREFIXES = ("另外", "顺便", "再", "那")
_STANDALONE_PREFIXES = (
    "什么是",
    "请问",
    "请查询",
    "查询",
    "帮我",
    "分析",
    "比较",
    "对比",
    "研究",
    "解释",
    "介绍",
    "列出",
    "总结",
)
_DEPENDENT_FINANCE_NOUNS = (
    "营收",
    "营业收入",
    "净利润",
    "利润",
    "毛利率",
    "净利率",
    "市盈率",
    "市净率",
    "估值",
    "现金流",
    "分红",
    "股价",
    "业绩",
    "表现",
    "风险",
    "情况",
    "数据",
    "结果",
)
_SOCIAL_OR_ACK_QUERIES = frozenset(
    {"再见", "谢谢", "感谢", "好的", "好", "明白了", "知道了", "收到"}
)
_TEMPORAL_DEMONSTRATIVE_PATTERN = re.compile(
    r"这个(?:季度|月|年度|财年|年份|阶段|期间)"
)
_CONDITION_ONLY_PATTERN = re.compile(
    r"(?:19|20)\d{2}年?|"
    r"(?:去年|今年|前年|明年|上年|本年)|"
    r"(?:第?[一二三四1234]季度|Q[1-4])|"
    r"(?:同比|环比|累计|单季|全年|年初至今)",
    re.IGNORECASE,
)
_COMPARISON_ENTITY_PATTERN = re.compile(
    r"(?:比较|对比)(?P<left>[\u4e00-\u9fffA-Za-z0-9]{2,20})"
    r"(?:和|与|、)(?P<right>[\u4e00-\u9fffA-Za-z0-9]{2,20})"
)
_ACTION_ENTITY_PATTERN = re.compile(
    r"(?:分析|研究|查询|查看|评估|介绍)"
    r"(?P<entity>[\u4e00-\u9fffA-Za-z0-9]{2,24})"
)
_ENTITY_PREFIXES = (
    "请帮我",
    "请",
    "帮我",
    "一下",
    "分析",
    "研究",
    "查询",
    "查看",
    "评估",
    "比较",
    "对比",
)
_ENTITY_STOP_WORDS = frozenset(
    {
        "上一轮问题",
        "上一轮回答",
        "某公司",
        "这家公司",
        "该公司",
        "它",
        "这个",
        "那个",
    }
)
_CLARIFICATION_CANCEL_QUERIES = frozenset(
    {"算了", "取消", "不用了", "不问了", "换个问题"}
)
_PENDING_MAX_TURNS = 2


@dataclass(frozen=True, slots=True)
class RewriteDecision:
    """确定性前置判断；只有 rewrite 才允许调用模型。"""

    action: RewriteAction
    reasons: tuple[RewriteReasonCode, ...] = ()


def _unique_strings(items: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in items if item.strip()))


def _latest_user_query(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _format_recent_dialogue(messages: list[AnyMessage]) -> str:
    """格式化近期对话；最后一条用户消息由 prompt 单独给出，此处排除。"""
    if not messages:
        return "无"

    last_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break

    prior = messages[:last_human_index] if last_human_index >= 0 else messages
    prior = prior[-_RECENT_MESSAGE_LIMIT:]
    if not prior:
        return "无"

    lines: list[str] = []
    for message in prior:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "无"


def _normalized_query(query: str) -> str:
    return "".join(query.split()).rstrip("？?！!。")


def _strip_discourse_prefix(query: str) -> str:
    result = query.lstrip("，,：:")
    for prefix in _DISCOURSE_PREFIXES:
        if result.startswith(prefix):
            return result[len(prefix):].lstrip("，,：:")
    return result


def _singular_ambiguous_markers(query: str) -> tuple[str, ...]:
    normalized = _TEMPORAL_DEMONSTRATIVE_PATTERN.sub("", query)
    return tuple(
        marker for marker in _SINGULAR_AMBIGUOUS_MARKERS if marker in normalized
    )


def _context_reference_markers(query: str) -> tuple[str, ...]:
    normalized = _TEMPORAL_DEMONSTRATIVE_PATTERN.sub("", query)
    return tuple(marker for marker in _CONTEXT_REFERENCE_MARKERS if marker in normalized)


def _looks_self_contained(query: str) -> bool:
    """识别即使带“另外/再”等口语连接词也能独立理解的问题。"""
    if query in _SOCIAL_OR_ACK_QUERIES:
        return True
    if _singular_ambiguous_markers(query):
        return False
    if query.startswith(_ALWAYS_DEPENDENT_PREFIXES):
        return False

    candidate = _strip_discourse_prefix(query)
    action_candidate = candidate.removeprefix("请").removeprefix("帮我")
    if candidate.startswith(_STANDALONE_PREFIXES) or action_candidate.startswith(
        _STANDALONE_PREFIXES
    ):
        return True
    if "什么是" in candidate or candidate.endswith("是什么"):
        return True

    for noun in _DEPENDENT_FINANCE_NOUNS:
        index = candidate.find(noun)
        if index >= 2:
            subject = candidate[:index]
            if (
                not _context_reference_markers(subject)
                and not _CONDITION_ONLY_PATTERN.fullmatch(subject)
            ):
                return True

    for suffix in _SHORT_FOLLOWUP_SUFFIXES:
        if not candidate.endswith(suffix):
            continue
        subject = candidate[: -len(suffix)].rstrip("是有")
        condition_with_metric = any(
            subject.endswith(noun)
            and bool(_CONDITION_ONLY_PATTERN.fullmatch(subject[: -len(noun)]))
            for noun in _DEPENDENT_FINANCE_NOUNS
        )
        if (
            len(subject) >= 2
            and subject not in _DEPENDENT_FINANCE_NOUNS
            and not any(noun == subject for noun in _DEPENDENT_FINANCE_NOUNS)
            and not _CONDITION_ONLY_PATTERN.fullmatch(subject)
            and not condition_with_metric
        ):
            return True
    return False


def _active_entity_candidates(state: FinAgentState) -> tuple[str, ...]:
    """只读取活跃话题实体；暂停话题不能制造当前问题歧义。"""
    summary = parse_summary_v2(state.get("conversation_summary_v2"))
    if summary is None or summary.active_topic_id is None:
        return ()
    topic = next(
        (item for item in summary.topics if item.topic_id == summary.active_topic_id),
        None,
    )
    if topic is None:
        return ()
    names = (
        list(topic.finance_context.securities)
        if topic.finance_context and topic.finance_context.securities
        else [item.name for item in topic.entities]
    )
    return tuple(dict.fromkeys(name for name in names if name.strip()))


def _clean_entity_candidate(value: str) -> str:
    candidate = value.strip(" ，,。？?：:；;的")
    for prefix in _ENTITY_PREFIXES:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):].strip(" ，,的")
    candidate = re.sub(r"(?:19|20)\d{2}年?", "", candidate)
    candidate = re.sub(r"(?:去年|今年|前年|明年|上年|本年)", "", candidate)
    candidate = candidate.removeprefix("那")
    candidate = candidate.strip(" ，,。？?：:；;的")
    if (
        len(candidate) < 2
        or candidate in _ENTITY_STOP_WORDS
        or candidate in _DEPENDENT_FINANCE_NOUNS
        or _CONDITION_ONLY_PATTERN.fullmatch(candidate)
        or _context_reference_markers(candidate)
    ):
        return ""
    return candidate


def _entities_from_text(text: str) -> tuple[str, ...]:
    """只从用户原文中的明确结构抽取候选，避免把模型回答当作事实来源。"""
    normalized = _normalized_query(text)
    found: list[str] = []
    comparison = _COMPARISON_ENTITY_PATTERN.search(normalized)
    if comparison:
        found.extend([comparison.group("left"), comparison.group("right")])

    for match in _ACTION_ENTITY_PATTERN.finditer(normalized):
        value = match.group("entity")
        cut_at = len(value)
        for marker in (*_DEPENDENT_FINANCE_NOUNS, "最近", "去年", "今年"):
            index = value.find(marker)
            if index >= 2:
                cut_at = min(cut_at, index)
        found.append(value[:cut_at])

    for noun in _DEPENDENT_FINANCE_NOUNS:
        index = normalized.find(noun)
        if index < 2:
            continue
        subject = normalized[:index]
        subject = re.split(r"[，,。；;？?]", subject)[-1]
        found.append(subject)

    cleaned = [_clean_entity_candidate(item) for item in found]
    return _unique_strings(cleaned)


def _context_entity_candidates(
    state: FinAgentState,
    messages: list[AnyMessage],
) -> tuple[str, ...]:
    candidates = list(_active_entity_candidates(state))
    last_human_index = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )
    prior = messages[:last_human_index] if last_human_index >= 0 else messages
    for message in prior[-_RECENT_MESSAGE_LIMIT:]:
        if not isinstance(message, HumanMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        candidates.extend(_entities_from_text(content))
    return _unique_strings(candidates)


def _pending_clarification(state: FinAgentState) -> dict[str, object]:
    raw = state.get("pending_query_clarification")
    return dict(raw) if isinstance(raw, Mapping) else {}


def build_pending_clarification(
    original_query: str,
    *,
    missing_fields: list[str] | None = None,
    candidate_entities: tuple[str, ...] = (),
    asked_question: str = "请补充缺失信息后重新提交。",
    revision: int = 1,
    remaining_turns: int = _PENDING_MAX_TURNS,
) -> dict[str, object]:
    """构造跨轮最小澄清状态；不保存消息历史或执行现场。"""
    digest = hashlib.sha256(
        f"{original_query}|{revision}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": "1.0",
        "clarification_id": f"query-{digest}",
        "kind": "query_rewrite",
        "original_query": original_query[:1000],
        "missing_fields": list(missing_fields or ["query_target"]),
        "candidate_entities": list(candidate_entities[:8]),
        "asked_question": asked_question[:500],
        "revision": max(1, revision),
        "remaining_turns": max(0, min(remaining_turns, _PENDING_MAX_TURNS)),
    }


def _rewrite_resolution(
    *,
    source: str,
    candidates: tuple[str, ...],
    original_query: str,
    rewritten_query: str,
) -> dict[str, object]:
    unresolved = list(_context_reference_markers(rewritten_query))
    resolved = [item for item in candidates if item in rewritten_query]
    return {
        "source": source,
        "candidate_entities": list(candidates),
        "resolved_entities": resolved,
        "unresolved_references": unresolved,
        "had_context_reference": bool(_context_reference_markers(original_query)),
        "confidence": 1.0 if not unresolved and (resolved or not candidates) else 0.0,
    }


def _validate_rewrite(
    original_query: str,
    rewritten_query: str,
    candidates: tuple[str, ...],
) -> tuple[bool, dict[str, object]]:
    resolution = _rewrite_resolution(
        source="context_rewrite",
        candidates=candidates,
        original_query=original_query,
        rewritten_query=rewritten_query,
    )
    if resolution["unresolved_references"]:
        return False, resolution
    if candidates and not resolution["resolved_entities"]:
        return False, resolution
    return True, resolution


async def _invoke_rewrite_model(
    human_prompt: str,
    config: RunnableConfig | None,
) -> str:
    result = await get_router_llm().ainvoke(
        [
            ("system", REWRITE_SYSTEM_PROMPT),
            ("human", human_prompt),
        ],
        config=config,
    )
    content = result.content if isinstance(result.content, str) else str(result.content)
    return content.strip()


def _parse_uncertain_clarification(content: str) -> str:
    if not content.startswith("__UNCERTAIN__"):
        return ""
    clarification = content.removeprefix("__UNCERTAIN__").strip()
    return clarification[:1800]


def classify_rewrite_need(
    state: FinAgentState,
    query: str,
    *,
    existing_summary: str,
    recent_dialogue: str,
    candidate_entities: tuple[str, ...] = (),
) -> RewriteDecision:
    """按问题完整性和上下文充分性返回唯一三态。"""
    normalized_query = _normalized_query(query)
    if not normalized_query:
        return RewriteDecision("passthrough", ("empty_query",))

    has_context = bool(existing_summary.strip()) or recent_dialogue.strip() not in {"", "无"}
    reasons: list[str] = []
    context_markers = _context_reference_markers(normalized_query)
    if context_markers:
        reasons.append("context_reference")
    if normalized_query.startswith(_FOLLOWUP_PREFIXES):
        reasons.append("followup_prefix")
    if (
        len(normalized_query) <= _MAX_SHORT_FOLLOWUP_LENGTH
        and normalized_query.endswith(_SHORT_FOLLOWUP_SUFFIXES)
    ):
        reasons.append("short_followup")

    if not reasons or _looks_self_contained(normalized_query):
        return RewriteDecision("passthrough", tuple(reasons or ["self_contained"]))
    if not has_context:
        return RewriteDecision("uncertain", (*reasons, "context_missing"))

    entities = candidate_entities or _active_entity_candidates(state)
    if not entities:
        return RewriteDecision("uncertain", (*reasons, "context_irrelevant"))
    if (
        len(entities) > 1
        and _singular_ambiguous_markers(normalized_query)
    ):
        return RewriteDecision("uncertain", (*reasons, "multiple_entity_candidates"))
    return RewriteDecision("rewrite", tuple(reasons))


async def query_rewrite_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """将本轮用户问题改写为完整问句，写入 rewritten_query。"""
    # 下轮开场兜底：若上轮 final_answer 未清掉，先丢掉残值再写本轮结果。
    stale = str(state.get("rewritten_query") or "").strip()
    if stale:
        logger.warning(
            "query_rewrite found uncleared rewritten_query, clearing before rewrite: {}",
            stale[:80],
        )

    history = list(state.get("messages") or [])
    query = _latest_user_query(history)
    if not query:
        return {
            "rewritten_query": "",
            "rewrite_status": "passthrough",
            "rewrite_reason_codes": ["empty_query"],
            "rewrite_failure_kind": "",
            "rewrite_resolution": {},
            "rewrite_clarification_message": "",
            "pending_query_clarification": {},
            "steps": ["query_rewrite:empty"],
        }

    existing_summary = project_conversation_context(state, purpose="rewrite")
    recent_dialogue = _format_recent_dialogue(history)
    candidates = _context_entity_candidates(state, history)
    pending = _pending_clarification(state)
    pending_original = str(pending.get("original_query") or "").strip()
    pending_turns = int(pending.get("remaining_turns") or 0) if pending else 0
    normalized_query = _normalized_query(query)

    if normalized_query in _CLARIFICATION_CANCEL_QUERIES:
        return {
            "rewritten_query": query,
            "rewrite_status": "passthrough",
            "rewrite_reason_codes": ["clarification_cancelled"],
            "rewrite_failure_kind": "",
            "rewrite_resolution": _rewrite_resolution(
                source="clarification_cancelled",
                candidates=(),
                original_query=query,
                rewritten_query=query,
            ),
            "rewrite_clarification_message": "",
            "pending_query_clarification": {},
            "steps": ["query_rewrite:clarification_cancelled"],
        }

    if (
        pending_original
        and pending_turns <= 0
        and not _looks_self_contained(normalized_query)
    ):
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "rewrite_reason_codes": ["clarification_expired"],
            "rewrite_failure_kind": "semantic",
            "rewrite_resolution": {},
            "rewrite_clarification_message": "",
            "pending_query_clarification": {},
            "steps": ["query_rewrite:uncertain:clarification_expired"],
        }

    pending_reply = bool(
        pending_original
        and pending_turns > 0
        and not _looks_self_contained(normalized_query)
        and len(normalized_query) <= 80
    )
    effective_query = query
    if pending_reply:
        effective_query = f"上一轮问题：{pending_original}\n用户本轮补充：{query}"
        pending_candidates = pending.get("candidate_entities") or []
        supplied_entity = _clean_entity_candidate(query)
        candidates = _unique_strings(
            [
                *candidates,
                *[str(item) for item in pending_candidates],
                *([supplied_entity] if supplied_entity else []),
            ]
        )

    decision = classify_rewrite_need(
        state,
        pending_original if pending_reply else query,
        existing_summary=existing_summary,
        recent_dialogue=recent_dialogue,
        candidate_entities=candidates,
    )
    if pending_reply:
        decision = RewriteDecision("rewrite", (*decision.reasons, "clarification_reply"))
    logger.info(
        "query_rewrite decision={} reasons={}",
        decision.action,
        decision.reasons,
    )
    if decision.action == "passthrough":
        return {
            "rewritten_query": query,
            "rewrite_status": "passthrough",
            "rewrite_reason_codes": list(decision.reasons),
            "rewrite_failure_kind": "",
            "rewrite_resolution": _rewrite_resolution(
                source="passthrough",
                candidates=(),
                original_query=query,
                rewritten_query=query,
            ),
            "rewrite_clarification_message": "",
            "pending_query_clarification": {},
            "steps": ["query_rewrite:passthrough"],
        }
    if decision.action == "uncertain":
        human = REWRITE_HUMAN_PROMPT.format(
            current_date=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
            existing_summary=existing_summary or "无",
            recent_dialogue=recent_dialogue,
            candidate_entities="、".join(candidates) or "无",
            pending_clarification=pending_original or "无",
            query=query,
            decision_guidance=(
                "规则已经确定当前问题不能安全改写。不得输出改写后的问题；"
                f"请使用格式二生成澄清。原因代码仅供理解：{', '.join(decision.reasons)}"
            ),
        )
        clarification_message = ""
        try:
            clarification_message = _parse_uncertain_clarification(
                await _invoke_rewrite_model(human, config)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "query_rewrite clarification generation failed: {}",
                type(exc).__name__,
            )
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "rewrite_reason_codes": list(decision.reasons),
            "rewrite_failure_kind": "semantic",
            "rewrite_resolution": _rewrite_resolution(
                source="rule_uncertain",
                candidates=candidates,
                original_query=query,
                rewritten_query=query,
            ),
            "rewrite_clarification_message": clarification_message,
            "pending_query_clarification": build_pending_clarification(
                query,
                candidate_entities=candidates,
            ),
            "steps": ["query_rewrite:uncertain"],
        }

    human = REWRITE_HUMAN_PROMPT.format(
        current_date=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        existing_summary=existing_summary or "无",
        recent_dialogue=recent_dialogue,
        candidate_entities="、".join(candidates) or "无",
        pending_clarification=pending_original or "无",
        query=effective_query,
        decision_guidance="当前问题允许尝试补全。能够可靠补全时使用格式一；否则使用格式二。",
    )

    try:
        rewritten = await _invoke_rewrite_model(human, config)
    except Exception:
        logger.exception("query_rewrite failed, mark uncertain")
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "rewrite_reason_codes": [*decision.reasons, "model_error"],
            "rewrite_failure_kind": "provider",
            "rewrite_resolution": _rewrite_resolution(
                source="provider_error",
                candidates=candidates,
                original_query=query,
                rewritten_query=query,
            ),
            "rewrite_clarification_message": "",
            "pending_query_clarification": (
                {**pending, "remaining_turns": max(0, pending_turns - 1)}
                if pending_reply
                else build_pending_clarification(query, candidate_entities=candidates)
            ),
            "steps": ["query_rewrite:uncertain:model_error"],
        }

    if rewritten == "__UNCERTAIN__" or rewritten.startswith("__UNCERTAIN__"):
        logger.info("query_rewrite uncertain, keep original query")
        clarification_message = _parse_uncertain_clarification(rewritten)
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "rewrite_reason_codes": [*decision.reasons, "model_uncertain"],
            "rewrite_failure_kind": "semantic",
            "rewrite_resolution": _rewrite_resolution(
                source="model_uncertain",
                candidates=candidates,
                original_query=query,
                rewritten_query=query,
            ),
            "rewrite_clarification_message": clarification_message,
            "pending_query_clarification": (
                {**pending, "remaining_turns": max(0, pending_turns - 1)}
                if pending_reply
                else build_pending_clarification(query, candidate_entities=candidates)
            ),
            "steps": ["query_rewrite:uncertain"],
        }

    if not rewritten:
        rewritten = query

    valid, resolution = _validate_rewrite(
        pending_original if pending_reply else query,
        rewritten,
        candidates,
    )
    if not valid:
        return {
            "rewritten_query": query,
            "rewrite_status": "uncertain",
            "rewrite_reason_codes": [*decision.reasons, "rewrite_validation_failed"],
            "rewrite_failure_kind": "semantic",
            "rewrite_resolution": resolution,
            "rewrite_clarification_message": "",
            "pending_query_clarification": (
                {**pending, "remaining_turns": max(0, pending_turns - 1)}
                if pending_reply
                else build_pending_clarification(query, candidate_entities=candidates)
            ),
            "steps": ["query_rewrite:uncertain:validation_failed"],
        }

    logger.info(
        "query_rewrite original={} rewritten={}",
        query[:80],
        rewritten[:80],
    )
    return {
        "rewritten_query": rewritten,
        "rewrite_status": "rewrite" if rewritten != query else "passthrough",
        "rewrite_reason_codes": list(decision.reasons),
        "rewrite_failure_kind": "",
        "rewrite_resolution": resolution,
        "rewrite_clarification_message": "",
        "pending_query_clarification": {},
        "steps": ["query_rewrite"],
    }
