"""纯规则查询约束解析器，不调用大模型。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

from retrieval.core.filters import MetadataFilters, query_infer_allowed_keys, supported_filter_keys

_YEAR_RE = re.compile(r"(?<![\dA-Za-z_-])(20\d{2})(?!\d)")
_TICKER_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_DOC_ID_RE = re.compile(
    r"(?<![A-Z0-9])(PDF-[A-Z0-9][A-Z0-9_-]*)(?![A-Z0-9])",
    re.IGNORECASE,
)
_REPORT_CONTEXT_RE = re.compile(r"(?:年报|年度报告|财报|财务报告|报告期|研报|研究报告)")
_YEAR_REPORT_HINT_RE = re.compile(r"(?:营业收入|净利润|研发费用|毛利率|资产负债|现金流|收入|费用|利润|财务指标)")
_PREDICTION_YEAR_RE = re.compile(r"(?:预计|预测|目标|到|将于|以后|之后)\s*(?:在)?20\d{2}年")
_HISTORICAL_CUTOFF_YEAR_RE = re.compile(r"截至\s*20\d{2}年?(?:底|年底|年末|年终)")

_ROOT = Path(__file__).resolve().parents[2]
_ENTITY_DICTIONARY_PATH = _ROOT / "retrieval" / "pdf_entity_dictionary.yaml"
_PDF_MANIFEST_PATH = _ROOT / "knowledge" / "raw" / "manifest_pdf.yaml"


@lru_cache(maxsize=1)
def _entity_entries() -> tuple[tuple[str, str, str], ...]:
    """加载可配置别名及 manifest 实体；返回 (alias, field, value)。"""
    entries: list[tuple[str, str, str]] = []

    def add(alias: Any, field: Any, value: Any) -> None:
        alias_text = str(alias or "").strip()
        value_text = str(value or "").strip()
        if alias_text and value_text and field in {"ticker", "issuer", "source", "doc_id"}:
            entries.append((alias_text, str(field), value_text))

    for path in (_ENTITY_DICTIONARY_PATH, _PDF_MANIFEST_PATH):
        if not path.exists():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if path == _ENTITY_DICTIONARY_PATH:
            for item in raw.get("entities", []) if isinstance(raw, dict) else []:
                if not isinstance(item, dict):
                    continue
                field = item.get("field")
                value = item.get("value")
                for alias in item.get("aliases", []):
                    add(alias, field, value)
        else:
            documents = raw.get("documents", {}) if isinstance(raw, dict) else {}
            for items in documents.values() if isinstance(documents, dict) else []:
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    for field in ("ticker", "issuer"):
                        value = item.get(field)
                        add(value, field, value)
                    doc_id = str(item.get("doc_id") or "").strip().upper()
                    if doc_id:
                        add(doc_id, "doc_id", doc_id)
                        # 文档标题和显式别名优先归一化为稳定的 doc_id。
                        add(item.get("title"), "doc_id", doc_id)
                        for alias in item.get("aliases", []) or []:
                            add(alias, "doc_id", doc_id)
                    # source 在索引中保存为文件名；文件名仍作为 source 的规范值。
                    source_value = Path(str(item.get("file") or "")).name
                    if source_value:
                        add(source_value, "source", source_value)

    # 长别名优先，避免“证券”之类短词覆盖具体机构名。
    unique = {(alias.casefold(), field, value): (alias, field, value) for alias, field, value in entries}
    return tuple(sorted(unique.values(), key=lambda item: len(item[0]), reverse=True))


def reload_query_entity_dictionary() -> None:
    """清空实体词典缓存，便于热更新配置后立即生效。"""
    _entity_entries.cache_clear()


def normalize_query_entity_filters(
    filters: MetadataFilters,
    *,
    knowledge_bases: list[str] | None = None,
) -> MetadataFilters:
    """将 LLM 候选实体归一化为 manifest 中的规范值；无法唯一映射的 source 丢弃。"""
    allowed = supported_filter_keys(knowledge_bases or None)
    entries = _entity_entries()
    normalized: MetadataFilters = {}
    for field, raw_value in (filters or {}).items():
        if field not in allowed or raw_value in (None, "", [], ()):
            continue
        value = str(raw_value).strip()
        if field not in {"doc_id", "issuer", "source", "ticker"}:
            normalized[field] = raw_value
            continue
        value_folded = value.casefold()
        candidates: list[tuple[int, str, str]] = []
        for alias, entry_field, canonical in entries:
            alias_folded = alias.casefold()
            if entry_field == field and (alias_folded == value_folded or alias_folded in value_folded):
                candidates.append((len(alias), alias, canonical))
        if field == "source":
            # source 的自然语言值只有在能指向具体文档时才升级为 doc_id。
            doc_candidates = [
                (length, alias, canonical)
                for alias, entry_field, canonical in entries
                for length in [len(alias)]
                if entry_field == "doc_id"
                and (alias.casefold() == value_folded or alias.casefold() in value_folded)
            ]
            if doc_candidates:
                _, _, canonical = max(doc_candidates, key=lambda item: item[0])
                normalized["doc_id"] = canonical
            elif candidates:
                _, _, canonical = max(candidates, key=lambda item: item[0])
                normalized["source"] = canonical
            continue
        if candidates:
            _, _, canonical = max(candidates, key=lambda item: item[0])
            normalized[field] = canonical
    return normalized


@dataclass(frozen=True)
class QueryConstraint:
    field: str
    value: Any
    operator: str
    role: str
    source: str
    confidence: float


@dataclass(frozen=True)
class StructuredQueryPlan:
    filters: MetadataFilters
    constraints: tuple[QueryConstraint, ...]
    soft_constraints: tuple[QueryConstraint, ...]
    unresolved: tuple[str, ...]
    reason: str


class RuleQueryConstraintParser:
    """从明确文本中提取安全的元数据约束；不猜测实体别名和年份语义。"""

    def parse(
        self,
        query: str,
        *,
        knowledge_bases: list[str] | None = None,
        user_filters: MetadataFilters | None = None,
    ) -> StructuredQueryPlan:
        text = str(query or "").strip()
        categories = [str(value) for value in (knowledge_bases or []) if value]
        infer_allowed = query_infer_allowed_keys(categories or None)
        allowed = supported_filter_keys(categories or None)
        filters: MetadataFilters = {}
        constraints: list[QueryConstraint] = []
        unresolved: list[str] = []

        # 用户筛选项是显式约束，优先级高于文本推断，但不能绕过 KB 合同。
        for field, value in (user_filters or {}).items():
            if field not in allowed or value in (None, "", [], ()):
                continue
            if field == "doc_id":
                value = str(value).strip().upper()
                if not re.fullmatch(r"PDF-[A-Z0-9][A-Z0-9_-]*", value, flags=re.IGNORECASE):
                    continue
            if field == "ticker":
                value = str(value).strip().upper()
                if not re.fullmatch(r"(?:\d{6}|[A-Z]{2,8})", value):
                    continue
            if field == "year":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if not 1900 <= value <= 2100:
                    continue
            filters[field] = value
            constraints.append(QueryConstraint(field, value, "eq", "user", "explicit", 1.0))

        doc_match = _DOC_ID_RE.search(text)
        if doc_match and "doc_id" in infer_allowed and "doc_id" not in filters:
            value = doc_match.group(1).upper()
            filters["doc_id"] = value
            constraints.append(QueryConstraint("doc_id", value, "eq", "document", "explicit", 1.0))

        ticker_match = _TICKER_RE.search(text)
        if ticker_match and "ticker" in infer_allowed and "ticker" not in filters:
            value = ticker_match.group(1).upper()
            filters["ticker"] = value
            constraints.append(QueryConstraint("ticker", value, "eq", "entity", "explicit", 1.0))

        year_match = _YEAR_RE.search(text)
        if year_match and "year" not in filters:
            year = int(year_match.group(1))
            report_context = bool(_REPORT_CONTEXT_RE.search(text))
            report_kb = any(category in {"annual_reports", "research_reports"} for category in categories)
            metric_context = bool(_YEAR_REPORT_HINT_RE.search(text))
            prediction_context = bool(_PREDICTION_YEAR_RE.search(text))
            historical_fact_context = bool(_HISTORICAL_CUTOFF_YEAR_RE.search(text))
            if "year" in infer_allowed and not prediction_context and (not historical_fact_context or report_context) and (report_context or report_kb or metric_context):
                filters["year"] = year
                constraints.append(
                    QueryConstraint("year", year, "eq", "reporting_year", "explicit", 0.98)
                )
            elif not report_context and not report_kb:
                unresolved.append("year_role")

        # 只把词典中的完整实体映射为硬过滤；未命中的自然语言实体交给 LLM。
        matched_entity_fields: set[str] = set()
        for alias, field, value in _entity_entries():
            if field not in allowed or field in filters or (field == "ticker" and "doc_id" in filters):
                continue
            if field in matched_entity_fields:
                continue
            if alias.casefold() not in text.casefold():
                continue
            filters[field] = value
            constraints.append(QueryConstraint(field, value, "eq", "entity", "dictionary", 0.99))
            matched_entity_fields.add(field)

        return StructuredQueryPlan(
            filters=filters,
            constraints=tuple(constraints),
            soft_constraints=(),
            unresolved=tuple(unresolved),
            reason="rule_constraints" if filters else "rule_empty",
        )


_default_parser = RuleQueryConstraintParser()


def parse_query_constraints(
    query: str,
    *,
    knowledge_bases: list[str] | None = None,
    user_filters: MetadataFilters | None = None,
) -> StructuredQueryPlan:
    return _default_parser.parse(
        query,
        knowledge_bases=knowledge_bases,
        user_filters=user_filters,
    )
