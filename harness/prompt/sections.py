"""稳定 system sections。"""

from __future__ import annotations

from dataclasses import dataclass

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
    "每一轮必须调用 submit_answer 才能结束。"
    "闲聊、澄清用 mode=direct；带数字的金融结论必须用 mode=grounded 并引用本轮 tool/result 中的 evidence_id。"
    "禁止把金融数字写进 direct_answer。"
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


def default_sections() -> tuple[PromptSection, ...]:
    return (
        PromptSection("identity", 10, IDENTITY_SECTION),
        PromptSection("compliance", 20, COMPLIANCE_SECTION),
        PromptSection("tool_discipline", 30, TOOL_DISCIPLINE_SECTION),
        PromptSection("skill_catalog", 40, skill_catalog_text()),
    )
