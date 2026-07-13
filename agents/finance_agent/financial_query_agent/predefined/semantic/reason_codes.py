"""coverage 结论的机器码与展示文案映射。"""

from __future__ import annotations

from typing import Literal

CoverageReasonCode = Literal[
    "METRIC_SEMANTICS_MISSING",
    "METRIC_UNRECOGNIZED",
    "COMPANY_METRIC_MAPPING_MISSING",
    "ANNUAL_DATA_NOT_FOUND",
    "QUARTER_ONLY_NO_ANNUAL",
    "PARTIAL_COMPARE_MISSING_ANNUAL",
    "GRANULARITY_CLARIFY_NEEDED",
]

REASON_TEMPLATES: dict[CoverageReasonCode, str] = {
    "METRIC_SEMANTICS_MISSING": "未识别到有效指标语义",
    "METRIC_UNRECOGNIZED": "无法识别指标：{requested_metric}",
    "COMPANY_METRIC_MAPPING_MISSING": "未找到任何公司级指标映射",
    "ANNUAL_DATA_NOT_FOUND": "未找到可用年报数据",
    "QUARTER_ONLY_NO_ANNUAL": (
        "白名单 predefined 仅支持年报口径；"
        "当前询问的年报指标仅存在季度数据、缺少年报事实，"
        "因此转交 text_to_sql 处理。"
    ),
    "PARTIAL_COMPARE_MISSING_ANNUAL": "部分公司缺少可比年报数据",
    "GRANULARITY_CLARIFY_NEEDED": "存在多种可用口径，需确认查询粒度",
}

_CLARIFY_REASON_CODES = frozenset(
    {"GRANULARITY_CLARIFY_NEEDED", "PARTIAL_COMPARE_MISSING_ANNUAL"}
)

# 兼容旧 import 路径
QUARTER_ONLY_FALLBACK_REASON = REASON_TEMPLATES["QUARTER_ONLY_NO_ANNUAL"]


def render_coverage_reasons(
    code: CoverageReasonCode | None,
    *,
    requested_metric: str = "",
) -> tuple[str, str]:
    """将 reason_code 渲染为 (clarify_reason, unavailable_reason)。"""
    if code is None:
        return "", ""
    text = REASON_TEMPLATES[code].format(requested_metric=requested_metric)
    if code in _CLARIFY_REASON_CODES:
        return text, ""
    return "", text


__all__ = [
    "CoverageReasonCode",
    "QUARTER_ONLY_FALLBACK_REASON",
    "REASON_TEMPLATES",
    "render_coverage_reasons",
]
