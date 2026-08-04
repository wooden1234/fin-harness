"""Main DeepAgent 的工具 Evidence 适配与财报事实冲突检查。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from agents.main_deep_agent.middleware.budget import TOOL_SOURCE_FAMILY
from agents.main_deep_agent.middleware.authorization import normalize_entity
from agents.orchestrator.contracts import Evidence

_FISCAL_PATTERNS = (
    re.compile(r"\b(?:FY\s*)?(20\d{2})\s*Q([1-4])\b", re.I),
    re.compile(r"\bQ([1-4])\s*(20\d{2})\b", re.I),
    re.compile(r"(20\d{2})\s*(?:财年)?第?([一二三四1-4])季度"),
)
_DATE_PATTERN = re.compile(r"\b(20\d{2})[-/年](0?[1-9]|1[0-2])[-/月](0?[1-9]|[12]\d|3[01])日?\b")
_EN_MONTH_DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s*(20\d{2})\b",
    re.I,
)
_PERCENT_PATTERN = re.compile(
    r"(?<!\d)(-?\d+(?:\.\d+)?)\s*(?:%|percent\b)",
    re.I,
)
_USD_PATTERN = re.compile(r"(?:\$\s*(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)\s*美元)")
_FINANCIAL_NUMBER_PATTERN = re.compile(
    r"(?:\$\s*-?\d|\d+(?:\.\d+)?\s*(?:%|percent\b)|\d+(?:\.\d+)?\s*(?:亿|万|million|billion)|营收|收入|净利润|profit|revenue)",
    re.I,
)
# 仅允许这些通用 IR/新闻前缀挂在注册官网上；任意消费子域（podcasts/music 等）不视为官方。
_OFFICIAL_HOST_LABELS = frozenset({"investor", "ir", "about", "newsroom", "press"})
_MONTH_INDEX = {
    name.lower(): index
    for index, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}
_DISCLAIMER_MARKERS = (
    "forward-looking",
    "前瞻性",
    "免责声明",
    "safe harbor",
    "非公认会计",
    "non-gaap",
)
# 通用披露状态线索（不绑定具体公司）。
_PREVIEW_CUES = (
    "业绩前瞻", "前瞻报告", "前瞻和重要关注", "业绩预告", "前瞻",
    "预计", "预估", "有望", "一致预期", "券商预期",
    "即将发布", "即将于", "即将公布", "待公布", "待发布",
    "earnings preview", "consensus estimate", "expected to report", "preview",
)
_ACTUAL_CUES = (
    "公告", "公布了", "今日公布", "今日电", "正式发布",
    "reported", "announces", "announced",
    "announced financial results", "quarterly results", "results for its",
    "earnings release",
)
_METRIC_MARKERS = {
    "aws_revenue_growth": ("AWS", "Amazon Web Services"),
    "services_revenue_growth": ("Services", "服务收入", "服务业务"),
    "iphone_revenue_growth": ("iPhone",),
    "advertising_revenue_growth": ("advertising", "广告业务", "广告收入"),
    "net_income_growth": ("net income", "净利润"),
    "eps_consensus": ("EPS estimate", "EPS consensus", "每股收益预期", "一致预期"),
    "revenue_growth": ("revenue grew", "revenue growth", "营收增长", "收入增长", "同比增长"),
}


def _publisher_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return hostname


def _domain_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def _entity_token(entity: str) -> str:
    """从实体名提取可用于域名匹配的字母数字 token，无公司白名单。"""
    return re.sub(r"[^a-z0-9]+", "", normalize_entity(entity).lower())


def _is_official_host(entity: str, hostname: str) -> bool:
    """启发式判断官网：二级域含实体 token，且非子域消费站（如 podcasts.）。

    无公司域名白名单；约 like aboutamazon.com / apple.com / newsroom.apple.com。
    """
    host = (hostname or "").lower().removeprefix("www.")
    token = _entity_token(entity)
    if len(token) < 3 or not host or "." not in host:
        return False
    labels = host.split(".")
    if len(labels) < 2:
        return False
    apex = labels[-2]
    brand_match = apex == token or (len(token) >= 4 and token in apex)
    if not brand_match:
        return False
    if len(labels) == 2:
        return True
    return labels[0] in _OFFICIAL_HOST_LABELS


def _entity_mentioned(entity: str, text: str) -> bool:
    """实体名需近似词边界命中；不做别名表扩展。"""
    alias = normalize_entity(entity)
    if not alias or len(alias) < 2:
        return False
    # 中文实体通常会紧邻财年、季度或数字，不能用英文单词边界限制。
    if re.search(r"[\u4e00-\u9fff]", alias):
        return alias in text
    pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
    return bool(re.search(pattern, text, re.I))


def _foreign_entity_dominant(requested: str, text: str) -> bool:
    """无公司注册表时不做跨实体污染推断；交给「未提及目标实体」规则处理。"""
    del requested, text
    return False


def _has_financial_numbers(text: str) -> bool:
    return bool(_FINANCIAL_NUMBER_PATTERN.search(text))


def _looks_like_disclaimer_only(text: str) -> bool:
    lowered = text.lower()
    if _has_financial_numbers(text):
        return False
    return any(marker in lowered for marker in _DISCLAIMER_MARKERS) or (
        "forward-looking statements" in lowered
        or "前瞻性陈述" in lowered
    )


def _is_forward_looking_piece(text: str) -> bool:
    """用语言线索判断是否为未披露前瞻/预估，而非已发布业绩。"""
    lowered = text.lower()
    preview_hits = sum(
        1 for cue in _PREVIEW_CUES if cue.lower() in lowered or cue in text
    )
    actual_hits = sum(
        1 for cue in _ACTUAL_CUES if cue.lower() in lowered or cue in text
    )
    if preview_hits <= 0:
        return False
    if ("即将" in text and ("发布" in text or "公布" in text)) or "待发布" in text or "待公布" in text:
        return True
    if actual_hits <= 0:
        return True
    return preview_hits >= actual_hits


def _source_grade(entity: str, hostname: str, text: str) -> str:
    canonical = normalize_entity(entity)
    if _is_official_host(canonical, hostname):
        if not _entity_mentioned(canonical, text) or not _has_financial_numbers(text):
            return "secondary"
        return "official"
    if _domain_matches(hostname, "sec.gov") and _entity_mentioned(canonical, text):
        return "official"
    return "secondary"


def _resolve_result_entity(requested_entity: str, hostname: str, text: str) -> str:
    """无公司注册表时保留请求实体；comparison 占位也不做域名反查。"""
    del hostname, text
    return normalize_entity(requested_entity)


def _fiscal_period(text: str) -> str:
    for index, pattern in enumerate(_FISCAL_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        first, second = match.groups()
        if index == 1:
            quarter, year = first, second
        else:
            year, quarter = first, second
        quarter = {"一": "1", "二": "2", "三": "3", "四": "4"}.get(quarter, quarter)
        return f"FY{year} Q{quarter}"
    ordinal = re.search(
        r"\b(20\d{2})(?:'s)?\s+(first|second|third|fourth)\s+fiscal\s+quarter\b",
        text,
        re.I,
    )
    if ordinal:
        quarter = {"first": 1, "second": 2, "third": 3, "fourth": 4}[ordinal.group(2).lower()]
        return f"FY{ordinal.group(1)} Q{quarter}"
    fiscal_year_first = re.search(
        r"\b(?:fiscal\s+)?(20\d{2})\s+(first|second|third|fourth)\s+quarter\b",
        text,
        re.I,
    )
    if fiscal_year_first:
        quarter = {"first": 1, "second": 2, "third": 3, "fourth": 4}[
            fiscal_year_first.group(2).lower()
        ]
        return f"FY{fiscal_year_first.group(1)} Q{quarter}"
    quarter_year = re.search(
        r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter.{0,80}?\b(20\d{2})\b",
        text,
        re.I,
    )
    if quarter_year:
        quarter = {"first": 1, "second": 2, "third": 3, "fourth": 4}[
            quarter_year.group(1).lower()
        ]
        return f"FY{quarter_year.group(2)} Q{quarter}"
    year_only = re.search(r"\b(20\d{2})\s*(?:年报|年度报告|annual report)\b", text, re.I)
    if year_only:
        return f"FY{year_only.group(1)} Q4"
    return ""


def _explicit_date(text: str) -> str:
    match = _DATE_PATTERN.search(text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    en_match = _EN_MONTH_DATE_PATTERN.search(text)
    if not en_match:
        return ""
    month, day, year = en_match.groups()
    return f"{year}-{_MONTH_INDEX[month.lower()]:02d}-{int(day):02d}"


def _period_end(text: str) -> str:
    """提取财季截止日；无法可靠解析时保持为空。"""
    match = re.search(
        r"(?:quarter|period)\s+ended\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(\d{1,2}),\s*(20\d{2})",
        text,
        re.I,
    )
    if not match:
        return ""
    month, day, year = match.groups()
    return f"{year}-{_MONTH_INDEX[month.lower()]:02d}-{int(day):02d}"


def _available_fields(
    text: str,
    *,
    official: bool,
    fiscal_period: str,
    is_preview: bool = False,
) -> list[str]:
    lowered = text.lower()
    fields = {"source_document"}
    if fiscal_period:
        fields.add("fiscal_period")
    has_numbers = _has_financial_numbers(text)
    if is_preview:
        fields.add("earnings_preview")
    elif (
        official
        and has_numbers
        and any(
            marker in lowered
            for marker in ("earnings", "results", "financial results", "财报", "业绩", "年报", "公告")
        )
    ):
        fields.add("official_earnings")
    has_growth = bool(_PERCENT_PATTERN.search(text)) and any(
        marker in lowered
        for marker in ("grew", "growth", "increased", "rose", "同比", "增长", "增速")
    )
    if has_growth and (
        is_preview
        or any(
            marker.lower() in lowered
            for markers in _METRIC_MARKERS.values()
            for marker in markers
        )
    ):
        # 预估稿常有「游戏/广告同比」，不强制已知 metric 词典。
        fields.add("segment_growth")
    if any(
        marker in lowered
        for marker in (
            "beat", "exceed", "超预期", "高于预期", "低于预期", "不及预期",
            "一致预期", "预估", "eps estimate", "consensus",
        )
    ):
        fields.add("expectation_comparison")
    if any(
        marker in lowered
        for marker in ("guidance", "outlook", "指引", "展望", "下季", "下一季", "下半年")
    ):
        fields.add("forward_guidance")
    return sorted(fields)


def _financial_facts(text: str, *, entity: str, fiscal_period: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    sentences = [item.strip() for item in re.split(r"[。！？\n]", text) if item.strip()]
    for sentence in sentences:
        percentages = _PERCENT_PATTERN.findall(sentence)
        lowered = sentence.lower()
        for metric, markers in _METRIC_MARKERS.items():
            if not any(marker.lower() in lowered for marker in markers):
                continue
            if metric == "eps_consensus":
                currency_match = _USD_PATTERN.search(sentence)
                if not currency_match:
                    continue
                value = next(
                    float(item)
                    for item in currency_match.groups()
                    if item is not None
                )
                unit = "currency/share"
                currency = "USD"
            else:
                if not percentages:
                    continue
                value = float(percentages[0])
                unit = "%"
                currency = ""
            facts.append(
                {
                    "entity": normalize_entity(entity),
                    "metric": metric,
                    "fiscal_period": fiscal_period,
                    "value": value,
                    "unit": unit,
                    "currency": currency,
                    "text": sentence[:500],
                }
            )
            break
    return facts


def _compact_display_text(
    *,
    title: str,
    snippet: str,
    fiscal_period: str,
    facts: list[dict[str, Any]],
) -> str:
    """生成面向用户的短要点，避免整段英文通稿灌进答案。"""
    if facts:
        lines = []
        for fact in facts[:3]:
            text = " ".join(str(fact.get("text") or "").split())
            if text:
                lines.append(text[:180])
        if lines:
            prefix = f"{fiscal_period}：" if fiscal_period else ""
            return prefix + "；".join(lines)
    # 优先截取含增速/营收的短句
    for sentence in re.split(r"[。！？\n\.]", snippet):
        cleaned = " ".join(sentence.split())
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if _PERCENT_PATTERN.search(cleaned) or any(
            marker in lowered
            for marker in ("revenue", "aws", "services", "营收", "增长", "operating income")
        ):
            head = f"{title}（{fiscal_period}）" if fiscal_period else title
            return f"{head}：{cleaned[:180]}"
    head = f"{title}（{fiscal_period}）" if fiscal_period else title
    return f"{head}：{snippet[:160]}"


def web_evidence_from_payload(
    data: Any,
    *,
    entity: str,
    entities: Sequence[str] | None = None,
) -> list[Evidence]:
    """将每条 Web 结果转换为独立 Evidence；聚合 answer 永不作为证据。"""
    if not isinstance(data, Mapping):
        return []
    observed_at = datetime.now(UTC).isoformat()
    evidence: list[Evidence] = []
    for raw in list(data.get("results") or []):
        if not isinstance(raw, Mapping):
            continue
        title = str(raw.get("title") or "").strip()
        url = str(raw.get("url") or "").strip()
        snippet = " ".join(str(raw.get("content") or "").split())[:2000]
        if not (title or url) or not snippet:
            continue
        hostname = _publisher_domain(url)
        # 归属与抽数以正文为准，避免标题平台名（如 * Podcasts）或 URL 子域误匹配。
        body_text = snippet
        target_entities = [
            normalize_entity(str(item).strip())
            for item in list(entities or [])
            if str(item).strip()
        ]
        if not target_entities and entity:
            target_entities = [normalize_entity(entity)]
        matched_entity = next(
            (
                candidate
                for candidate in target_entities
                if _entity_mentioned(candidate, body_text)
            ),
            "",
        )
        resolved_entity = matched_entity or _resolve_result_entity(
            entity,
            hostname,
            body_text,
        )
        entity_mismatch = bool(
            target_entities
            and (
                not matched_entity
                or _foreign_entity_dominant(resolved_entity, body_text)
            )
        )
        extraction_failed = _looks_like_disclaimer_only(body_text)
        is_preview = (not entity_mismatch) and _is_forward_looking_piece(
            f"{title}\n{body_text}"
        )
        grade = "secondary" if entity_mismatch or extraction_failed or is_preview else _source_grade(
            resolved_entity, hostname, body_text
        )
        period = _fiscal_period(f"{title}\n{body_text}")
        published_at = str(raw.get("published_date") or "").strip() or _explicit_date(
            f"{title}\n{body_text}"
        )
        if not period and published_at:
            period = _fiscal_period(f"{title}\n{body_text} {published_at}")
        period_end = str(raw.get("period_end") or "").strip() or _period_end(
            f"{title}\n{body_text}"
        )
        fields = (
            []
            if entity_mismatch or extraction_failed
            else _available_fields(
                f"{title}\n{body_text}",
                official=grade == "official",
                fiscal_period=period,
                is_preview=is_preview,
            )
        )
        facts = (
            []
            if entity_mismatch or extraction_failed or is_preview
            else _financial_facts(
                f"{title}\n{body_text}",
                entity=resolved_entity,
                fiscal_period=period,
            )
        )
        display_text = _compact_display_text(
            title=title,
            snippet=snippet,
            fiscal_period=period if not entity_mismatch else "",
            facts=facts,
        )
        if is_preview and display_text:
            display_text = f"【未披露·前瞻/预估】{display_text}"
        digest = hashlib.sha256(
            f"{url}|{title}|{snippet}".encode("utf-8")
        ).hexdigest()[:20]
        evidence.append(
            Evidence(
                evidence_id=f"web:{digest}",
                task_id="main",
                source_type="web.search",
                provider=hostname or "web",
                title=title or hostname or "联网来源",
                content=snippet,
                url=url or None,
                published_at=published_at or None,
                observed_at=observed_at,
                confidence=(
                    0.55 if is_preview
                    else 0.9 if grade == "official"
                    else 0.4 if entity_mismatch
                    else 0.7
                ),
                metadata={
                    "entity": resolved_entity,
                    "publisher_domain": hostname,
                    "source_grade": grade,
                    "document_type": (
                        "earnings_preview" if is_preview
                        else "earnings_release" if "official_earnings" in fields
                        else "web_article"
                    ),
                    "fiscal_period": period if not entity_mismatch else "",
                    "period_end": period_end,
                    "available_fields": fields,
                    "data_time_valid": bool(published_at),
                    "display_text": display_text,
                    "displayable": not entity_mismatch and not extraction_failed,
                    "entity_mismatch": entity_mismatch,
                    "extraction_failed": extraction_failed,
                    "is_preview": is_preview,
                    "facts": facts,
                },
            )
        )
    return evidence


def detect_fact_conflicts(evidence: list[Evidence]) -> list[dict[str, Any]]:
    """按实体、指标、财季和单位识别数值冲突并确定可否自动消解。"""
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        for raw in list(item.metadata.get("facts") or []):
            if not isinstance(raw, Mapping):
                continue
            key = (
                str(raw.get("entity") or ""),
                str(raw.get("metric") or ""),
                str(raw.get("fiscal_period") or ""),
                str(raw.get("unit") or ""),
                str(raw.get("currency") or ""),
            )
            grouped[key].append(
                {
                    **dict(raw),
                    "evidence_id": item.evidence_id,
                    "source_grade": item.metadata.get("source_grade"),
                    "published_at": item.published_at,
                }
            )

    conflicts: list[dict[str, Any]] = []
    for key, facts in grouped.items():
        values = sorted({float(item["value"]) for item in facts})
        if len(values) <= 1:
            continue
        metric = key[1]
        period = key[2]
        # 空财季上的多值不拉黑 evidence，避免年报长文误杀。
        if not period:
            conflicts.append(
                {
                    "key": key,
                    "values": values,
                    "evidence_ids": [],
                    "resolved": True,
                    "resolution": "period_unknown",
                    "preferred_evidence_id": "",
                }
            )
            continue
        official = [item for item in facts if item.get("source_grade") == "official"]
        rounding = key[3] == "%" and max(values) - min(values) <= 1.0
        consensus = "consensus" in metric or "expectation" in metric
        resolved = bool(official) and not consensus
        resolution = (
            "official_source_preferred"
            if resolved
            else "rounding_difference"
            if rounding
            else "scope_difference"
            if consensus
            else "unresolved"
        )
        conflicts.append(
            {
                "key": key,
                "values": values,
                "evidence_ids": [item["evidence_id"] for item in facts],
                "resolved": resolved or rounding,
                "resolution": resolution,
                "preferred_evidence_id": official[0]["evidence_id"] if resolved else "",
            }
        )
    return conflicts


__all__ = ["detect_fact_conflicts", "web_evidence_from_payload"]

def _safe_json(value: Any, *, limit: int = 12000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


def _bounded_tool_payload(value: Any, *, max_chars: int = 32000) -> Any:
    """限制单次工具回填体积，避免大表挤占 Main Agent 上下文。"""
    serialized = _safe_json(value, limit=max_chars + 1)
    if len(serialized) <= max_chars:
        return value
    return {
        "truncated": True,
        "preview": serialized[:max_chars],
        "notice": "工具结果已按上下文预算截断，请缩小查询范围后重试以查看更多。",
    }


def _find_data_time(value: Any, *, depth: int = 0) -> str | None:
    """从结构化工具结果中提取数据自身时间，避免把抓取时间当作数据时点。"""
    if depth > 3:
        return None
    if isinstance(value, Mapping):
        for key in (
            "published_at",
            "as_of",
            "date",
            "report_date",
            "data_date",
            "updated_at",
            "time",
        ):
            raw = value.get(key)
            if raw not in (None, ""):
                if key == "time" and isinstance(raw, (int, float)):
                    try:
                        return datetime.fromtimestamp(raw, tz=UTC).isoformat()
                    except (OverflowError, OSError, ValueError):
                        continue
                return str(raw)
        for nested in value.values():
            found = _find_data_time(nested, depth=depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value[:10]:
            found = _find_data_time(nested, depth=depth + 1)
            if found:
                return found
    return None


_IWENCAI_FACT_QUERY_TOOL_IDS = frozenset({
    "iwencai.query",
    "iwencai.finance.query",
})
_STRUCTURED_EVIDENCE_TOOL_IDS = frozenset({
    "knowledge.fact.lookup",
    "finance.fact.lookup",
    "finance.query_advanced",
    "iwencai.query",
    "iwencai.finance.query",
    "iwencai.screen",
    "iwencai.usstock.screen",
    "iwencai.rating.query",
    "iwencai.market.query",
    "iwencai.report.search",
    "iwencai.announcement.search",
    "iwencai.industry.query",
})
_OFFICIAL_STRUCTURED_TOOL_IDS = frozenset({
    "knowledge.fact.lookup",
    "finance.fact.lookup",
    "finance.query_advanced",
    "iwencai.announcement.search",
})
_STRUCTURED_GRADE_TOOL_IDS = frozenset({
    "iwencai.query",
    "iwencai.finance.query",
    "iwencai.screen",
    "iwencai.usstock.screen",
    "iwencai.market.query",
})


def _unwrap_tool_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """问财等工具常把业务字段包在 data 下；合并顶层与内层以便抽取。"""
    nested = data.get("data")
    if not isinstance(nested, Mapping):
        return dict(data)
    merged = dict(nested)
    for key, value in data.items():
        if key == "data":
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _row_display_chunk(row: Mapping[str, Any], *, max_fields: int = 8) -> str:
    chunks: list[str] = []
    for key, value in row.items():
        if value in (None, "", [], {}):
            continue
        key_text = str(key).strip()
        if not key_text or key_text.startswith("_"):
            continue
        chunks.append(f"{key_text}={value}")
        if len(chunks) >= max_fields:
            break
    return "，".join(chunks)


def _format_preview_cell(value: object) -> str:
    if value in (None, ""):
        return "—"
    text = " ".join(str(value).split()).strip()
    return text[:80] if len(text) > 80 else text


def _is_numeric_cell(value: object) -> bool:
    if value in (None, ""):
        return False
    try:
        float(str(value).replace(",", "").replace("%", "").strip())
        return True
    except ValueError:
        return False


def _preview_keys_for_datas(rows: list[Mapping[str, Any]]) -> list[str]:
    """按「数值列优先、保持首行出现顺序」选列，不写死字段名白名单。"""
    first = rows[0]
    keys = [
        str(key).strip()
        for key in first
        if str(key).strip() and not str(key).startswith("_")
    ]
    if not keys:
        return []

    def score(key: str) -> tuple[int, int, int]:
        numeric_hits = sum(1 for row in rows if _is_numeric_cell(row.get(key)))
        # 累计财务列优先；单季次之；最新价/涨跌幅尽量不进摘要。
        if "最新价" in key or "涨跌幅" in key:
            preference = 100
        elif "单季度" in key or "单季" in key:
            preference = 40
        elif "累计" in key and any(
            marker in key for marker in ("营业收入", "营业额", "收入", "净利润", "归母", "净利")
        ):
            preference = -100
        elif "累计" in key:
            preference = -40
        else:
            preference = 0
        return (preference, -numeric_hits, keys.index(key))

    return sorted(keys, key=score)[:8]


def _preview_table_from_datas(
    datas: object,
    *,
    query: str = "",
) -> dict[str, Any] | None:
    """把问财 datas 收成前端可渲染的小表，避免 key=value 长串。"""
    if not isinstance(datas, list):
        return None
    rows_raw = [item for item in datas[:5] if isinstance(item, Mapping)]
    if not rows_raw:
        return None
    columns = _preview_keys_for_datas(rows_raw)
    if not columns:
        return None
    rows = [
        [_format_preview_cell(item.get(column)) for column in columns]
        for item in rows_raw
    ]
    preview: dict[str, Any] = {"columns": columns, "rows": rows}
    cleaned_query = " ".join(str(query or "").split()).strip()[:200]
    if cleaned_query:
        preview["query"] = cleaned_query
    return preview


def _short_summary_from_preview(
    preview: Mapping[str, Any],
    *,
    entity: str = "",
) -> str:
    """用预览表前几列动态拼短摘要，不写死指标名。"""
    columns = list(preview.get("columns") or [])
    rows = list(preview.get("rows") or [])
    if not columns or not rows:
        return ""
    first = list(rows[0])
    pairs = [
        (str(columns[index]), str(first[index]))
        for index in range(min(len(columns), len(first)))
        if str(first[index]).strip() and str(first[index]).strip() != "—"
    ]
    if not pairs:
        return ""
    # 首个非纯数值列更像名称；其余取最多两个数值列
    name = entity
    for key, value in pairs:
        if not _is_numeric_cell(value):
            name = value
            break
    numeric_parts = [
        f"{key} {value}"
        for key, value in pairs
        if _is_numeric_cell(value)
    ][:2]
    parts = [part for part in (name, *numeric_parts) if part]
    return "，".join(parts)[:240]


def _display_text_from_mapping(data: Mapping[str, Any], *, limit: int = 800) -> str:
    for key in ("answer", "summary", "display_text", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            if text and not text.startswith(("{", "[")):
                return text[:limit]
    facts = data.get("facts")
    if isinstance(facts, list) and facts:
        parts: list[str] = []
        for item in facts[:5]:
            if not isinstance(item, Mapping):
                continue
            company = str(item.get("company") or item.get("entity") or "").strip()
            metric = str(item.get("metric") or "").strip()
            value = str(item.get("value") or "").strip()
            year = str(item.get("period_year") or item.get("fiscal_period") or "").strip()
            chunk = " ".join(part for part in (company, year, metric, value) if part)
            if chunk:
                parts.append(chunk)
        if parts:
            return "；".join(parts)[:limit]
    nested = data.get("data")
    if isinstance(nested, Mapping):
        nested_text = _display_text_from_mapping(nested, limit=limit)
        if nested_text:
            return nested_text
    datas = data.get("datas")
    if isinstance(datas, list) and datas:
        parts: list[str] = []
        query = str(data.get("query") or "").strip()
        prefix = f"问财「{query}」：" if query else ""
        for item in datas[:5]:
            if not isinstance(item, Mapping):
                continue
            chunk = _row_display_chunk(item)
            if chunk:
                parts.append(chunk)
        if parts:
            return f"{prefix}{'；'.join(parts)}"[:limit]
    return ""


def _structured_tool_evidence(tool_id: str, data: Mapping[str, Any]) -> list[Evidence]:
    """为财务/问财等结构化工具写入与 Web 对齐的覆盖字段。"""
    source_family = TOOL_SOURCE_FAMILY.get(tool_id, "unknown")
    observed_at = datetime.now(UTC).isoformat()
    payload = _unwrap_tool_mapping(data)
    datas = [
        dict(item)
        for item in list(payload.get("datas") or [])
        if isinstance(item, Mapping)
    ]
    if tool_id in _IWENCAI_FACT_QUERY_TOOL_IDS and not datas and not payload.get("facts"):
        return []
    if tool_id in _IWENCAI_FACT_QUERY_TOOL_IDS and len(datas) > 1:
        evidence: list[Evidence] = []
        for row in datas:
            row_payload = dict(payload)
            row_payload["datas"] = [row]
            # 多行返回中的顶层 facts 无可靠行归属，拆分时不跨实体复制。
            row_payload["facts"] = []
            evidence.extend(_structured_tool_evidence(tool_id, row_payload))
        return evidence
    display_payload = payload
    if tool_id in _IWENCAI_FACT_QUERY_TOOL_IDS and datas:
        display_payload = {
            "query": payload.get("query") or data.get("query") or "",
            "datas": datas,
        }
    display_text = _display_text_from_mapping(display_payload)
    if not display_text:
        # 保留顶层 query，便于空 facts 时仍有可读摘要。
        display_text = _display_text_from_mapping(data)
    content = display_text or _safe_json(display_payload, limit=4000)
    if not content or content in {"null", "{}", "[]"}:
        return []
    published_at = _find_data_time(payload) or _find_data_time(data)
    entity = ""
    for key in ("entity", "company"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            entity = normalize_entity(raw)
            break
    if not entity:
        companies = payload.get("companies") or payload.get("facts")
        if isinstance(companies, list):
            for item in companies:
                if isinstance(item, Mapping):
                    candidate = str(item.get("company") or item.get("entity") or "").strip()
                else:
                    candidate = str(item).strip()
                if candidate:
                    entity = normalize_entity(candidate)
                    break
    period = ""
    for key in ("fiscal_period", "period", "period_label"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            period = raw.strip()
            break
    if not period:
        year = payload.get("period_year") or payload.get("fiscal_year")
        if year not in (None, ""):
            period = f"FY{year} Q4"
        else:
            for item in list(payload.get("facts") or []):
                if not isinstance(item, Mapping):
                    continue
                year = item.get("period_year") or item.get("fiscal_year")
                if year not in (None, ""):
                    period = f"FY{year} Q4"
                    break
    lowered = content.lower()
    fields = {"source_document"}
    if period:
        fields.add("fiscal_period")
    grade = "secondary"
    if tool_id in _OFFICIAL_STRUCTURED_TOOL_IDS:
        grade = "structured"
        fields.add("official_earnings")
    elif tool_id in _STRUCTURED_GRADE_TOOL_IDS:
        grade = "structured"
    if tool_id == "iwencai.rating.query" or any(
        marker in lowered
        for marker in ("一致预期", "超预期", "eps", "预测", "consensus", "estimate")
    ):
        fields.add("expectation_comparison")
    if any(marker in lowered for marker in ("增长", "增速", "%", "同比", "growth")):
        fields.add("segment_growth")
    facts: list[dict[str, Any]] = []
    for item in list(payload.get("facts") or []):
        if not isinstance(item, Mapping):
            continue
        fact_entity = normalize_entity(
            str(item.get("company") or item.get("entity") or entity or "")
        )
        metric = str(
            item.get("canonical_metric") or item.get("metric") or ""
        ).strip()
        if not metric:
            continue
        try:
            value = float(str(item.get("value")).replace("%", "").replace(",", ""))
        except (TypeError, ValueError):
            continue
        fact_period = str(item.get("fiscal_period") or period or "").strip()
        if item.get("period_year") not in (None, ""):
            fact_period = f"FY{item.get('period_year')} Q4"
        facts.append(
            {
                "entity": fact_entity,
                "metric": metric,
                "fiscal_period": fact_period,
                "value": value,
                "unit": str(item.get("unit") or ""),
                "currency": str(item.get("currency") or ""),
                "text": display_text[:500],
            }
        )
    readable = bool(display_text) and not display_text.startswith(("{", "["))
    preview_table = _preview_table_from_datas(
        payload.get("datas") or data.get("datas"),
        query=str(payload.get("query") or data.get("query") or ""),
    )
    short_summary = ""
    if preview_table is not None:
        short_summary = _short_summary_from_preview(preview_table, entity=entity)
    digest = hashlib.sha256(f"{tool_id}:{content}".encode("utf-8")).hexdigest()[:20]
    metadata: dict[str, Any] = {
        "source_family": source_family,
        "tool_id": tool_id,
        "entity": entity,
        "publisher_domain": source_family,
        "source_grade": grade,
        "fiscal_period": period,
        "available_fields": sorted(fields),
        "data_time_valid": bool(published_at or period),
        # 面向用户的步骤卡优先用短摘要；完整 key=value 仍留在 content 供排查。
        "display_text": short_summary or display_text or content[:800],
        "displayable": readable,
        "facts": facts,
    }
    if preview_table is not None:
        metadata["preview_table"] = preview_table
    return [
        Evidence(
            evidence_id=f"main:{digest}",
            task_id="main",
            source_type=tool_id,
            provider=source_family,
            title=tool_id,
            content=content[:4000],
            url=str(payload.get("url") or data.get("url") or "") or None,
            published_at=published_at,
            observed_at=observed_at,
            confidence=0.85 if grade == "structured" else 0.75,
            metadata=metadata,
        )
    ]


def _pdf_catalog_evidence(data: Mapping[str, Any]) -> list[Evidence]:
    """把本地 PDF 目录结果转成可展示、可引用的 Evidence。"""
    documents = [
        item
        for item in list(data.get("documents") or [])
        if isinstance(item, Mapping) and str(item.get("doc_id") or "").strip()
    ]
    if not documents:
        return []
    by_category: dict[str, list[str]] = {}
    lines: list[str] = []
    for item in documents:
        doc_id = str(item.get("doc_id") or "").strip()
        title = str(item.get("title") or doc_id).strip()
        category = str(item.get("category") or "unknown").strip() or "unknown"
        by_category.setdefault(category, []).append(f"{doc_id}｜{title}")
        lines.append(f"{category}/{doc_id}: {title}")
    summary_parts = [
        f"{category} {len(items)} 份"
        for category, items in sorted(by_category.items())
    ]
    display_text = (
        f"本地 PDF 目录共 {len(documents)} 份："
        + "；".join(summary_parts)
    )
    content = "\n".join(lines)[:4000]
    digest = hashlib.sha256(f"knowledge.pdf.catalog:{content}".encode("utf-8")).hexdigest()[:20]
    observed_at = datetime.now(UTC).isoformat()
    return [
        Evidence(
            evidence_id=f"main:{digest}",
            task_id="main",
            source_type="knowledge.pdf.catalog",
            provider="knowledge",
            title="本地 PDF 目录",
            content=content,
            observed_at=observed_at,
            confidence=0.9,
            metadata={
                "source_family": "knowledge",
                "tool_id": "knowledge.pdf.catalog",
                "display_text": display_text,
                "displayable": True,
                "source_grade": "structured",
                "doc_ids": [str(item.get("doc_id") or "").strip() for item in documents],
                "available_fields": ["source_document"],
                "data_time_valid": True,
            },
        )
    ]


def _calculation_evidence(data: Mapping[str, Any]) -> list[Evidence]:
    """把批量计算结果拆成可独立引用、可追溯的 Calculation Evidence。"""
    observed_at = datetime.now(UTC).isoformat()
    evidence: list[Evidence] = []
    for raw in list(data.get("calculations") or []):
        if not isinstance(raw, Mapping) or not raw.get("ok"):
            continue
        formula = str(raw.get("formula") or "").strip()
        evidence_ids = [
            str(item).strip()
            for item in list(raw.get("input_evidence_ids") or [])
            if str(item).strip()
        ]
        if not formula or not evidence_ids:
            continue
        calculation_id = str(raw.get("calculation_id") or "calculation").strip()
        value = raw.get("value")
        unit = str(raw.get("unit") or "")
        display_text = f"{calculation_id}: {value}{unit}"
        content = _safe_json(raw, limit=3000)
        digest = hashlib.sha256(
            f"calculation.run:{content}".encode("utf-8")
        ).hexdigest()[:20]
        evidence.append(
            Evidence(
                evidence_id=f"main:{digest}",
                task_id="main",
                source_type="calculation.run",
                provider="calculation",
                title=calculation_id,
                content=content,
                observed_at=observed_at,
                confidence=1.0,
                metadata={
                    "source_family": "calculation",
                    "tool_id": "calculation.run",
                    "calculation_id": calculation_id,
                    "formula": formula,
                    "input_evidence_ids": evidence_ids,
                    "operand_refs": list(raw.get("operand_refs") or []),
                    "display_text": display_text,
                    "displayable": True,
                    "data_time_valid": True,
                    "facts": [],
                },
            )
        )
    return evidence


def _evidence_from_payload(tool_id: str, data: Any) -> list[Evidence]:
    """优先接收工具原生 Evidence，否则为成功的外部结果建立可追溯证据。"""
    observed_at = datetime.now(UTC).isoformat()
    evidence: list[Evidence] = []
    if isinstance(data, Mapping):
        for raw in list(data.get("evidence") or []):
            try:
                item = raw if isinstance(raw, Evidence) else Evidence.model_validate(raw)
            except ValueError:
                continue
            evidence.append(
                item.model_copy(
                    update={
                        "task_id": "main",
                        "metadata": {
                            **dict(item.metadata),
                            "source_family": TOOL_SOURCE_FAMILY.get(tool_id, "unknown"),
                            "data_time_valid": bool(item.published_at)
                            or bool(item.metadata.get("data_time_valid")),
                            "displayable": bool(item.metadata.get("displayable", True)),
                        },
                    }
                )
            )
        if evidence:
            return evidence

    if tool_id in {
        "knowledge.faq.search",
        "knowledge.pdf.search",
    }:
        return []
    if tool_id == "knowledge.pdf.catalog" and isinstance(data, Mapping):
        return _pdf_catalog_evidence(data)
    if tool_id == "calculation.run" and isinstance(data, Mapping):
        return _calculation_evidence(data)
    if tool_id in _STRUCTURED_EVIDENCE_TOOL_IDS and isinstance(data, Mapping):
        return _structured_tool_evidence(tool_id, data)

    content = _safe_json(data)
    if not content or content in {"null", "{}", "[]"}:
        return []
    source_family = TOOL_SOURCE_FAMILY.get(tool_id, "unknown")
    digest = hashlib.sha256(f"{tool_id}:{content}".encode("utf-8")).hexdigest()[:20]
    published_at = _find_data_time(data)
    url = None
    if isinstance(data, Mapping):
        url = str(data.get("url") or "") or None
    return [
        Evidence(
            evidence_id=f"main:{digest}",
            task_id="main",
            source_type=tool_id,
            provider=source_family,
            title=tool_id,
            content=content,
            url=url,
            published_at=published_at,
            observed_at=observed_at,
            confidence=0.8,
            metadata={
                "source_family": source_family,
                "tool_id": tool_id,
                "data_time_valid": bool(published_at),
                "displayable": False,
                **(
                    {
                        "input_evidence_ids": list(data.get("input_evidence_ids") or []),
                        "formula": str(data.get("formula") or ""),
                    }
                    if tool_id == "calculation.run" and isinstance(data, Mapping)
                    else {}
                ),
            },
        )
    ]


_COMPACT_EVIDENCE_METADATA_KEYS = (
    "entity",
    "fiscal_period",
    "display_text",
    "displayable",
    "source_grade",
    "facts",
    "calculation_id",
    "formula",
    "input_evidence_ids",
    "operand_refs",
)


def _select_web_evidence_for_model(
    evidence: list[Evidence],
) -> tuple[list[Evidence], bool]:
    """模型侧只保留可展示条目；全不可展示时保底 1 条。"""
    displayable = [item for item in evidence if item.metadata.get("displayable")]
    if displayable:
        return displayable, False
    if evidence:
        return [evidence[0]], True
    return [], False


def _compact_evidence_for_model(
    evidence: list[Evidence],
    *,
    tool_id: str = "",
) -> list[dict[str, Any]]:
    """供模型引用的精简 Evidence；去掉大字段 content，journal 仍保留完整对象。"""
    items = list(evidence)
    if tool_id == "web.search":
        items, _ = _select_web_evidence_for_model(items)
    compact: list[dict[str, Any]] = []
    for item in items:
        metadata = {
            key: item.metadata.get(key)
            for key in _COMPACT_EVIDENCE_METADATA_KEYS
            if key in item.metadata
        }
        compact.append(
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "url": item.url,
                "source_type": item.source_type,
                "provider": item.provider,
                "published_at": item.published_at,
                "metadata": metadata,
            }
        )
    return compact


def _sanitize_tool_payload(
    tool_id: str,
    data: Any,
    *,
    evidence: list[Evidence] | None = None,
) -> Any:
    """Web 回传用 displayable 摘要替代原文，避免 content 与 evidence 双份膨胀。"""
    if tool_id != "web.search" or not isinstance(data, Mapping):
        return _bounded_tool_payload(data)
    selected, displayable_fallback = _select_web_evidence_for_model(list(evidence or []))
    dropped_low_quality = max(0, len(evidence or []) - len(selected))
    if displayable_fallback:
        dropped_low_quality = max(0, len(evidence or []) - 1)
    results = []
    for item in selected:
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        summary = str(metadata.get("display_text") or item.title or "").strip()
        summary = " ".join(summary.split())[:800]
        results.append(
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "entity": metadata.get("entity") or "",
                "fiscal_period": metadata.get("fiscal_period") or "",
                "summary": summary,
                "url": item.url or "",
            }
        )
    cleaned = {
        "results": results,
        "answer_omitted": True,
        "dropped_low_quality": dropped_low_quality,
        "displayable_fallback": displayable_fallback,
    }
    score_stats = data.get("score_stats")
    if isinstance(score_stats, Mapping):
        cleaned["score_stats"] = dict(score_stats)
    return _bounded_tool_payload(cleaned)
