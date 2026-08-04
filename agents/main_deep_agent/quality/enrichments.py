"""终答增强：追问清洗与已核验多期图表（不编造序列、不写死问句模板）。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from agents.orchestrator.contracts import Evidence

_MAX_FOLLOW_UPS = 3
_MAX_FOLLOW_UP_CHARS = 80
_MAX_CHART_POINTS = 8

# 与质量门投资动作约束对齐的过滤词，只用于拦截，不用于生成问句。
_ACTION_FOLLOW_UP_MARKERS = (
    "适合投资",
    "买入",
    "卖出",
    "买卖点",
    "加仓",
    "减仓",
    "能不能买",
    "还能上车",
    "仓位",
    "目标价",
    "止损",
)


def sanitize_follow_ups(
    raw_follow_ups: object,
    *,
    query: str = "",
    investment_action_sensitive: bool = False,
) -> list[str]:
    """清洗模型给出的追问：去重、截断、过滤投资动作话术。不生成任何硬编码问句。"""
    if not isinstance(raw_follow_ups, list):
        return []
    query_compact = "".join(str(query or "").split())
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_follow_ups:
        text = " ".join(str(raw or "").split()).strip()
        if not text or len(text) > _MAX_FOLLOW_UP_CHARS:
            continue
        compact = "".join(text.split())
        if not compact or compact in seen:
            continue
        if query_compact and (compact in query_compact or query_compact in compact):
            continue
        if any(marker in text for marker in _ACTION_FOLLOW_UP_MARKERS):
            continue
        if investment_action_sensitive and any(
            marker in text for marker in ("投资", "买入", "卖出", "仓位")
        ):
            continue
        seen.add(compact)
        cleaned.append(text)
        if len(cleaned) >= _MAX_FOLLOW_UPS:
            break
    return cleaned


def _period_sort_key(period: str) -> tuple[int, int]:
    text = str(period or "")
    year_match = re.search(r"(20\d{2})", text)
    year = int(year_match.group(1)) if year_match else 0
    quarter = 4
    q_match = re.search(r"Q([1-4])", text, re.I)
    if q_match:
        quarter = int(q_match.group(1))
    elif "一季" in text:
        quarter = 1
    elif "半年" in text or "中报" in text:
        quarter = 2
    elif "三季" in text:
        quarter = 3
    return year, quarter


def _collect_fact_points(
    evidence: Sequence[Evidence],
    *,
    used_evidence_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(used_evidence_ids or [])
    points: list[dict[str, Any]] = []
    for item in evidence:
        if allowed and item.evidence_id not in allowed:
            continue
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        for fact in list(metadata.get("facts") or []):
            if not isinstance(fact, Mapping):
                continue
            entity = str(fact.get("entity") or metadata.get("entity") or "").strip()
            metric = str(fact.get("metric") or "").strip() or "value"
            period = str(
                fact.get("fiscal_period") or metadata.get("fiscal_period") or ""
            ).strip()
            if not entity or not period:
                continue
            try:
                value = float(fact.get("value"))
            except (TypeError, ValueError):
                continue
            # 展示名优先用事实原文中的可读字段，避免写死中文映射表。
            label = str(fact.get("label") or fact.get("metric_name") or metric).strip()
            points.append(
                {
                    "entity": entity,
                    "metric": metric,
                    "label": label,
                    "period": period,
                    "value": value,
                    "unit": str(fact.get("unit") or ""),
                    "currency": str(fact.get("currency") or ""),
                }
            )
    return points


def build_answer_charts(
    *,
    evidence: Sequence[Evidence],
    used_evidence_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """仅在已核验事实覆盖 ≥2 个期间时出图；否则返回空（降级不画）。"""
    points = _collect_fact_points(evidence, used_evidence_ids=used_evidence_ids)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[(point["entity"], point["metric"])].append(point)

    charts: list[dict[str, Any]] = []
    for (entity, metric), rows in grouped.items():
        by_period: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_period[row["period"]] = row
        ordered = sorted(
            by_period.values(), key=lambda item: _period_sort_key(item["period"])
        )
        if len(ordered) < 2:
            continue
        ordered = ordered[:_MAX_CHART_POINTS]
        categories = [str(item["period"]) for item in ordered]
        values = [float(item["value"]) for item in ordered]
        unit = next(
            (str(item.get("unit") or "") for item in ordered if item.get("unit")),
            "",
        )
        metric_label = str(ordered[0].get("label") or metric).strip() or metric
        is_ratio = unit == "%" or "growth" in metric.lower()
        title = f"{entity} {metric_label}".strip()
        chart: dict[str, Any] = {
            "type": "combo",
            "title": title,
            "categories": categories,
            "unit": unit or ("%" if is_ratio else ""),
        }
        if is_ratio:
            chart["lines"] = [{"name": metric_label, "values": values}]
            chart["bars"] = []
        else:
            chart["bars"] = [{"name": metric_label, "values": values}]
            chart["lines"] = []
        charts.append(chart)
        if len(charts) >= 2:
            break
    return charts


# 兼容旧导入名：生成逻辑已移除，仅保留清洗。
build_follow_ups = sanitize_follow_ups

__all__ = ["build_answer_charts", "build_follow_ups", "sanitize_follow_ups"]
