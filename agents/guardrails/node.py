"""Guardrails 节点：安全护栏校验（前置，纯规则，零 LLM 调用）"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.states import FinAgentState
from agents.turn_workspace import begin_turn_workspace
from app.core.logger import get_logger

logger = get_logger(service="guardrails")

# 注入关键词黑名单（纯规则，不依赖 LLM）
_INJECTION_PATTERNS = [
    r"忽略.*指令",
    r"ignore.*instruction",
    r"你.*现在.*是.*DAN",
    r"system\s*prompt",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"你的.*系统.*提示词",
    r"忘记.*之前",
    r"扮演.*角色",
    r"pretend.*you.*are",
]

# 敏感信息正则
_PII_PATTERNS = [
    (r"\d{15}(\d{2}[0-9Xx])?", "身份证号"),
    (r"\d{16}(\d{3})?", "银行卡号"),
    (r"1[3-9]\d{9}", "手机号"),
]

# 恶意内容关键词
_HARMFUL_KEYWORDS = ["自杀", "自残", "杀人", "爆炸", "枪支", "毒品", "色情"]


def _latest_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _check_injection(query: str) -> tuple[bool, str]:
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True, f"检测到注入模式: {pattern}"
    return False, ""


def _check_pii(query: str) -> tuple[bool, str]:
    for pattern, pii_type in _PII_PATTERNS:
        if re.search(pattern, query):
            return True, f"检测到{pii_type}"
    return False, ""


def _check_harmful(query: str) -> tuple[bool, str]:
    for kw in _HARMFUL_KEYWORDS:
        if kw in query:
            return True, f"检测到敏感词: {kw}"
    return False, ""


async def guardrails_node(
    state: FinAgentState,
    config: RunnableConfig = None,
) -> dict:
    """纯规则护栏校验，前置在 Supervisor 之前。

    每轮入口先 Overwrite 重置临时工作区（含 steps），避免跨轮累加。
    """
    workspace = begin_turn_workspace()
    query = _latest_user_query(list(state.get("messages") or []))
    if not query:
        return {**workspace, "guardrails_pass": True}

    for check_fn in [_check_injection, _check_pii, _check_harmful]:
        blocked, reason = check_fn(query)
        if blocked:
            logger.warning("guardrails blocked: {}", reason)
            return {
                **workspace,
                "guardrails_pass": False,
                "guardrails_reason": reason,
            }

    logger.info("guardrails passed")
    return {**workspace, "guardrails_pass": True}


def guardrails_edge(state: FinAgentState) -> str:
    """条件边：通过 → context_compressor，拦截 → final_answer"""
    if state.get("guardrails_pass", True):
        return "context_compressor"
    return "final_answer"
