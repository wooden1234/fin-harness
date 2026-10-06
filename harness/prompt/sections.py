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
    "规则概念用 faq-knowledge；财经新闻、政策动态、板块驱动用 news-search；"
    "通用网页或非财经来源用 web-search。"
    "禁止用网页、新闻或 FAQ 补报表数字。"
    "计算前先检查单位、币种、期间和口径；跨币种且用户未提供汇率或折算时点时，"
    "只能要求补充口径，不能自行假设汇率、举例计算或给出同比百分比。"
    "用户已明确给出多个可能实体且无法唯一确定时，先向用户澄清，身份明确前不要调用事实查询工具。"
    "用户给出的实体名称与证据中的主体全名精确匹配时，优先采用精确匹配；"
    "不要因为还存在名称更长的前缀相同主体而把精确匹配误判为歧义。"
    "把去年、今年、最新等相对时间按当前系统日期解析，调用工具时必须在 query 中写明解析后的绝对年份或日期，"
    "且最终答案的期间必须与工具请求一致。"
    "筛选条件中的小于、大于是严格边界，不得擅自改成小于等于或大于等于；必须排除等于阈值的记录。"
    "用户更正实体后，只回答更正后的实体；不要把更正前实体的数据附带为参考。"
    "用户消息中的密钥、令牌、密码等敏感值不得进入工具参数、日志或回答，"
    "提醒用户处理凭据时也只能称为‘该密钥’或‘该凭据’，不得逐字复述。"
    "互不依赖的查询（多家公司、多指标、估值与财务）必须在同一轮发出多个 tool_calls，系统会并行执行；"
    "禁止一家查完再查下一家。能一条问财问句覆盖的对比，优先合并成一次查询，"
    "或把公司列表放进 entities，由工具合成一次请求返回多行。"
    "用户要求全部、完整结果时，如果工具返回 next_cursor 或等价分页标记，"
    "必须在本轮携带该游标继续请求，不要询问用户是否继续；游标为空或达到工具调用上限时停止，"
    "若仍未取完则明确说明结果不完整。"
    "同一工具在本轮用户问题中最多调用 2 次（首次 + 重试 1 次）；禁止第三次再调。"
    "工具失败且没有更匹配的工具或技能时，直接用文字告知用户无法查到，"
    "不要改用公告、研报等不相关工具继续试。"
    "复杂问题（多公司对比、多指标拆解、多源综合、因果/驱动）收齐检索后："
    "工具列表里有 finalign_analyze 则调用一次并把返回正文作为最终回答；"
    "没有该工具时直接根据已检索结果用正文回答。"
    "不要在未检索时调用 finalign_analyze，也不要用它规划下一步工具。"
    "若调用后失败，不要改调其它 LLM 补偿，直接根据已检索材料用正文回答。"
    "查完或闲聊后直接用助手正文回复用户即可结束本轮，不要再调工具。"
    "不要在用户可见正文里写「数据来源」、工具名或 evidence_id；这些只留在工具结果中供核对。"
    "不要编造数据。"
)

MEMORY_TOOL_SECTION = (
    "用户明确说请记住、以后默认、改成或忘记某项回答偏好时，调用 memory_write 或 memory_delete。"
    "不要每轮先调这些工具。key 含糊时直接向用户确认，不要猜测删除哪一条。"
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


def preference_sections(
    preferences: Mapping[str, Any] | None = None,
    turn_overrides: Mapping[str, Any] | None = None,
    session_overrides: Mapping[str, Any] | None = None,
) -> tuple[PromptSection, ...]:
    """长期偏好、本会话声明与本轮覆盖，接在稳定 system 之后。

    身份 / 合规 / 工具纪律 / skill 目录保持固定前缀。
    长期偏好 order=50，本会话 order=55，本轮临时 order=60。
    """
    prefs = {key: value for key, value in dict(preferences or {}).items() if value is not None}
    session = {
        key: value for key, value in dict(session_overrides or {}).items() if value is not None
    }
    overrides = {
        key: value for key, value in dict(turn_overrides or {}).items() if value is not None
    }
    sections: list[PromptSection] = []
    if prefs:
        sections.append(
            PromptSection(
                "user_preferences",
                50,
                "[用户长期偏好]\n"
                f"{_preference_lines(prefs)}\n"
                "回答默认遵守这些长期偏好，并覆盖身份段的默认中文与简洁设定。"
                "用户提问时使用的语言或打招呼不改变回答方式。"
                "该偏好适用于所有用户可见输出，包括问候、闲聊、记忆查看、澄清、错误说明和工具结果总结。"
                "展示某项偏好时，也必须按有效的 response_language 回答，不能只展示而不执行。"
                "用户明确声明本次会话改用某种方式，或直接说用中文回答、用英文回答时，本会话后续都改用该方式。",
            )
        )
    if session:
        sections.append(
            PromptSection(
                "session_overrides",
                55,
                "[本会话要求]\n"
                f"{_preference_lines(session)}\n"
                "用户已声明本会话采用这些方式。本会话内按这里回答，并覆盖冲突的长期偏好。"
                "不要写入长期记忆。",
            )
        )
    if overrides:
        sections.append(
            PromptSection(
                "turn_overrides",
                60,
                "[本轮临时要求]\n"
                f"{_preference_lines(overrides)}\n"
                "这些要求只在当前轮生效，并覆盖冲突的长期偏好和本会话要求。",
            )
        )
    return tuple(sections)


def preference_section(
    preferences: Mapping[str, Any] | None = None,
    turn_overrides: Mapping[str, Any] | None = None,
    session_overrides: Mapping[str, Any] | None = None,
) -> PromptSection | None:
    """兼容单段调用：多段时拼成一段，order 取末尾。"""
    sections = preference_sections(preferences, turn_overrides, session_overrides)
    if not sections:
        return None
    if len(sections) == 1:
        return sections[0]
    return PromptSection(
        "user_preferences",
        sections[-1].order,
        "\n\n".join(item.text for item in sections),
    )


def default_sections() -> tuple[PromptSection, ...]:
    return (
        PromptSection("identity", 10, IDENTITY_SECTION),
        PromptSection("compliance", 20, COMPLIANCE_SECTION),
        PromptSection("tool_discipline", 30, TOOL_DISCIPLINE_SECTION),
        PromptSection("memory_tools", 32, MEMORY_TOOL_SECTION),
        PromptSection("skill_catalog", 40, skill_catalog_text()),
    )
