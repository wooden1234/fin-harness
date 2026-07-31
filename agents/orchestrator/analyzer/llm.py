"""Analyzer LLM 调用：结构化输出；总调用次数由节点限制为最多 2 次。"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from agents.llm import get_router_llm
from agents.orchestrator.analyzer.prompts import (
    ANALYZER_REPAIR_SYSTEM_PROMPT,
    build_analyzer_system_prompt,
)
from agents.orchestrator.analyzer.schema import AnalyzerInputEnvelope, AnalyzerOutput
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_analyzer")


def is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    name = type(exc).__name__.lower()
    markers = (
        "timeout",
        "connection",
        "ratelimit",
        "rate_limit",
        "apiconnection",
        "internalserver",
        "serviceunavailable",
        "429",
        "502",
        "503",
    )
    return any(marker in name for marker in markers)


async def ainvoke_analyzer(
    *,
    system_prompt: str,
    human_prompt: str,
    config: RunnableConfig | None,
) -> AnalyzerOutput:
    llm = get_router_llm()
    return cast(
        AnalyzerOutput,
        await llm.with_structured_output(
            AnalyzerOutput, method="json_mode"
        ).ainvoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ],
            config=config,
        ),
    )


async def analyze_once(
    envelope: AnalyzerInputEnvelope,
    config: RunnableConfig | None = None,
) -> AnalyzerOutput:
    """单次画像分析，不做内部重试。"""
    return await ainvoke_analyzer(
        system_prompt=build_analyzer_system_prompt(),
        human_prompt=envelope.model_dump_json(exclude_none=True),
        config=config,
    )


async def repair_profile(
    envelope: AnalyzerInputEnvelope,
    raw: AnalyzerOutput,
    issues: list[str],
    config: RunnableConfig | None = None,
) -> AnalyzerOutput:
    """单次修复调用，不做内部重试。"""
    payload = raw.model_dump_json()
    human_prompt = (
        f"输入信封：\n{envelope.model_dump_json(exclude_none=True)}\n\n"
        f"校验问题：{', '.join(issues)}\n\n"
        f"待修正输出：{payload}"
    )
    return await ainvoke_analyzer(
        system_prompt=ANALYZER_REPAIR_SYSTEM_PROMPT,
        human_prompt=human_prompt,
        config=config,
    )


__all__ = [
    "ainvoke_analyzer",
    "analyze_once",
    "is_transient_api_error",
    "repair_profile",
]
