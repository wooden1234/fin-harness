"""稳定 system sections。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from skills.loader import list_skill_catalog

IDENTITY_SECTION = (
    "你是小财，面向投资者的金融助手。"
    "用简洁中文回答。不知道就说不知道，不编造数据。"
)

COMPLIANCE_SECTION = (
    "不得给出个性化买卖指令，不得承诺收益、稳赚、必涨或保证收益。"
    "投资有风险，陈述必须可核对。"
)

TOOL_DISCIPLINE_SECTION = (
    "需要外部事实时先读 skill 目录再调用工具。"
    "当前财务指标默认 finance-query（同花顺官方库）；"
    "本地已入库年报当时披露用 local-financial-facts；"
    "规则概念用 faq-knowledge；新闻事件用 web-search。"
    "禁止用网页或 FAQ 补报表数字。"
    "互不依赖的查询（多家公司、多指标、估值与财务）必须在同一轮发出多个 tool_calls，系统会并行执行；"
    "禁止一家查完再查下一家。能一条问财问句覆盖的对比，优先合并成一次查询，"
    "或把公司列表放进 entities，由工具合成一次请求返回多行。"
    "同一工具在本轮用户问题中最多调用 2 次（首次 + 重试 1 次）；禁止第三次再调。"
    "工具失败且没有更匹配的工具或技能时，立即 submit_answer（mode=direct）告知用户无法查到，"
    "不要改用公告、研报等不相关工具继续试。"
    "每一轮必须调用 submit_answer 才能结束。"
    "闲聊、澄清用 mode=direct；带数字的金融结论必须用 mode=grounded 并引用本轮 tool/result 中的 evidence_id。"
    "禁止把金融数字写进 direct_answer。"
)

MEMORY_TOOL_SECTION = (
    "用户明确说请记住、以后默认、改成或忘记某项回答偏好时，调用 memory_write 或 memory_delete。"
    "不要每轮先调这些工具。key 含糊时用 submit_answer 向用户确认，不要猜测删除哪一条。"
)


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    order: int
    text: str


def skill_catalog_text() -> str:
    lines = ["可用 skill："]
    catalog = list_skill_catalog()
    if not catalog:
        lines.append("（目录为空）")
        return "\n".join(lines)
    for entry in catalog:
        description = entry.description.strip() or "（无描述）"
        lines.append(f"- {entry.name}: {description}")
    return "\n".join(lines)


def _preference_lines(values: Mapping[str, Any]) -> str:
    return "\n".join(f"- {key}={value}" for key, value in sorted(values.items()))


def preference_section(
    preferences: Mapping[str, Any] | None = None,
    turn_overrides: Mapping[str, Any] | None = None,
) -> PromptSection | None:
    """长期偏好与本轮覆盖；两者都空则不贡献 section。"""
    prefs = {key: value for key, value in dict(preferences or {}).items() if value is not None}
    overrides = {
        key: value for key, value in dict(turn_overrides or {}).items() if value is not None
    }
    blocks: list[str] = []
    if prefs:
        blocks.append(
            "[用户长期偏好]\n"
            f"{_preference_lines(prefs)}\n"
            "仅在当前请求未明确指定时参考长期偏好；当前轮用户要求优先。"
            "长期偏好中的语言与详略覆盖身份段的默认中文与简洁设定。"
        )
    if overrides:
        blocks.append(
            "[本轮临时要求]\n"
            f"{_preference_lines(overrides)}\n"
            "这些要求只在当前轮生效，并覆盖冲突的长期偏好。"
        )
    if not blocks:
        return None
    return PromptSection("user_preferences", 25, "\n\n".join(blocks))


def default_sections() -> tuple[PromptSection, ...]:
    return (
        PromptSection("identity", 10, IDENTITY_SECTION),
        PromptSection("compliance", 20, COMPLIANCE_SECTION),
        PromptSection("tool_discipline", 30, TOOL_DISCIPLINE_SECTION),
        PromptSection("memory_tools", 32, MEMORY_TOOL_SECTION),
        PromptSection("skill_catalog", 40, skill_catalog_text()),
    )
