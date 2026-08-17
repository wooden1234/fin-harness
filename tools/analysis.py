"""finalign 综合分析工具：多源检索之后成稿，不负责规划或检索。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from tools.core.base import ToolSpec
from tools.core.registry import register_tool


@tool(parse_docstring=True)
async def finalign_analyze(
    question: str,
    materials: list[str] | None = None,
) -> dict[str, Any]:
    """对已检索的多工具结果做金融综合分析，产出给用户的最终回答。

    复杂对比、多指标拆解、多源综合时，先并行查完数据再调用本工具。
    单指标直查或闲聊不要调用。

    Args:
        question: 用户原问题
        materials: 可选的工具结果摘要；留空时运行时从本轮 tool/result 收集
    """
    from capabilities.analysis import synthesize_answer

    return await synthesize_answer(question=question, materials=materials or [])


register_tool(
    ToolSpec(
        tool_id="finalign.analyze",
        name="finalign_analyze",
        description=(
            "复杂问题在收齐多个检索工具结果后调用：综合分析并写出最终答案。"
            "不要在未检索时调用；不要用来规划下一步工具。"
        ),
        risk_level="low",
        read_only=True,
        requires_human_approval=False,
        timeout_seconds=130.0,
    ),
    langchain_tool=finalign_analyze,
)


__all__ = ["finalign_analyze"]
