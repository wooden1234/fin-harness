"""Metadata filter engine for KB-scoped retrieval.

- Engine (matchers, metadata_matches): stable Python
- KB contracts (filter fields, query_infer_fields): knowledge/raw/kb_filter_profiles.yaml
- Document type routing: retrieval/retrievers/pdf_kb_router.py (LLM)
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

MetadataFilters = dict[str, Any]

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROFILES_PATH = _ROOT / "knowledge" / "raw" / "kb_filter_profiles.yaml"


class MatchMode(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"


@dataclass(frozen=True)
class FilterRule:
    filter_key: str
    mode: MatchMode
    alt_keys: tuple[str, ...] = ()
    metadata_fields: tuple[str, ...] = ()
    fuzzy_fields: tuple[str, ...] = ()
    matcher: Callable[[dict[str, Any], Any], bool] | None = None


@dataclass(frozen=True)
class InferFieldSpec:
    field_key: str
    pattern: re.Pattern[str]
    cast: str = "str"


@dataclass(frozen=True)
class KnowledgeBaseProfile:
    kb_id: str
    rules: tuple[FilterRule, ...]
    query_infer_fields: tuple[str, ...] = ()
    required_chunk_metadata: tuple[str, ...] = ()
    on_empty: str = "abstain"


@dataclass(frozen=True)
class FilterConfig:
    profiles: dict[str, KnowledgeBaseProfile]
    infer_fields: dict[str, InferFieldSpec]
    common_required_chunk_metadata: tuple[str, ...] = ()
    common_query_infer_fields: tuple[str, ...] = ()
    route_priority: tuple[str, ...] = ()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _filter_value(filters: MetadataFilters, rule: FilterRule) -> Any:
    for key in (rule.filter_key, *rule.alt_keys):
        value = filters.get(key)
        if value not in (None, "", [], ()):
            return value
    return None


def _match_exact_fields(metadata: dict[str, Any], expected: Any, fields: tuple[str, ...]) -> bool:
    target = _norm(expected)
    for field in fields:
        actual = metadata.get(field)
        if actual not in (None, ""):
            return _norm(actual) == target
    return False


def _match_category(metadata: dict[str, Any], expected: Any) -> bool:
    categories = [str(v) for v in _as_list(expected) if v]
    if not categories:
        return True
    meta_category = _norm(metadata.get("category"))
    return meta_category in {_norm(category) for category in categories}


def _match_year(metadata: dict[str, Any], expected: Any) -> bool:
    target = str(expected)
    fiscal = metadata.get("fiscal_year")
    if fiscal not in (None, ""):
        return str(fiscal) == target
    indexed = metadata.get("year")
    if indexed not in (None, ""):
        return str(indexed) == target
    effective_date = str(metadata.get("effective_date") or "")
    if len(effective_date) >= 4:
        return effective_date[:4] == target
    return False


def _match_fuzzy_contains(
    metadata: dict[str, Any],
    expected: Any,
    fields: tuple[str, ...],
) -> bool:
    needle = _norm(expected)
    return any(needle in _norm(metadata.get(field)) for field in fields)


def _match_company(metadata: dict[str, Any], expected: Any) -> bool:
    if _match_exact_fields(metadata, expected, ("company",)):
        return True
    return _match_fuzzy_contains(
        metadata,
        expected,
        fields=("company", "ticker", "issuer", "title", "source", "doc_group", "doc_id"),
    )


MATCHERS: dict[str, Callable[[dict[str, Any], Any], bool]] = {
    "category": _match_category,
    "year": _match_year,
    "company": _match_company,
}


def _apply_rule(metadata: dict[str, Any], rule: FilterRule, expected: Any) -> bool:
    if rule.matcher is not None:
        return rule.matcher(metadata, expected)
    if rule.mode is MatchMode.EXACT:
        return _match_exact_fields(metadata, expected, rule.metadata_fields)
    return _match_fuzzy_contains(metadata, expected, rule.fuzzy_fields)


def _parse_field_rule(name: str, spec: dict[str, Any]) -> FilterRule:
    mode = MatchMode(str(spec.get("mode", "exact")).lower())
    matcher_name = spec.get("matcher")
    matcher = MATCHERS.get(str(matcher_name)) if matcher_name else None
    metadata_fields = tuple(str(f) for f in _as_list(spec.get("metadata_fields")))
    fuzzy_fields = metadata_fields if mode is MatchMode.FUZZY and metadata_fields else tuple(
        str(f) for f in _as_list(spec.get("fuzzy_fields"))
    )
    if mode is MatchMode.FUZZY and not fuzzy_fields and not matcher:
        raise ValueError(f"field_rules.{name}: fuzzy rule needs metadata_fields or matcher")
    if mode is MatchMode.EXACT and not matcher and not metadata_fields:
        raise ValueError(f"field_rules.{name}: exact rule needs metadata_fields or matcher")
    return FilterRule(
        filter_key=name,
        mode=mode,
        alt_keys=tuple(str(k) for k in _as_list(spec.get("alt_filter_keys"))),
        metadata_fields=metadata_fields,
        fuzzy_fields=fuzzy_fields,
        matcher=matcher,
    )


def _parse_infer_fields(raw: dict[str, Any]) -> dict[str, InferFieldSpec]:
    infer_fields: dict[str, InferFieldSpec] = {}
    for field_key, spec in (raw.get("infer_fields") or {}).items():
        key = str(field_key)
        if isinstance(spec, str):
            infer_fields[key] = InferFieldSpec(
                field_key=key,
                pattern=re.compile(spec, flags=re.IGNORECASE),
            )
            continue
        pattern = re.compile(str(spec.get("pattern", "")), flags=re.IGNORECASE)
        infer_fields[key] = InferFieldSpec(
            field_key=key,
            pattern=pattern,
            cast=str(spec.get("cast") or "str"),
        )
    for field_key, pattern in (raw.get("infer_patterns") or {}).items():
        key = str(field_key)
        if key not in infer_fields:
            infer_fields[key] = InferFieldSpec(
                field_key=key,
                pattern=re.compile(str(pattern), flags=re.IGNORECASE),
            )
    return infer_fields


def load_filter_config(path: Path | None = None) -> FilterConfig:
    config_path = path or Path(os.getenv("KB_FILTER_PROFILES_PATH", str(DEFAULT_PROFILES_PATH)))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    field_rules = {
        name: _parse_field_rule(name, spec)
        for name, spec in (raw.get("field_rules") or {}).items()
    }
    common = [str(k) for k in _as_list(raw.get("common_filters"))]
    common_required = tuple(
        str(field) for field in _as_list(raw.get("common_required_chunk_metadata"))
    )
    common_query_infer = tuple(
        str(field) for field in _as_list(raw.get("common_query_infer_fields"))
    )

    profiles: dict[str, KnowledgeBaseProfile] = {}
    for kb_id, spec in (raw.get("knowledge_bases") or {}).items():
        filter_names = common + [str(k) for k in _as_list(spec.get("filters"))]
        rules: list[FilterRule] = []
        seen: set[str] = set()
        for name in filter_names:
            if name in seen:
                continue
            rule = field_rules.get(name)
            if rule is None:
                raise KeyError(f"knowledge_bases.{kb_id}: unknown filter {name!r}")
            rules.append(rule)
            seen.add(name)
        kb_required = tuple(str(f) for f in _as_list(spec.get("required_chunk_metadata")))
        kb_query_infer = tuple(str(f) for f in _as_list(spec.get("query_infer_fields")))
        profiles[kb_id] = KnowledgeBaseProfile(
            kb_id=kb_id,
            rules=tuple(rules),
            query_infer_fields=kb_query_infer,
            required_chunk_metadata=common_required + kb_required,
            on_empty=str(spec.get("on_empty") or "abstain"),
        )

    infer_fields = _parse_infer_fields(raw)

    route_priority = tuple(str(kb_id) for kb_id in _as_list(raw.get("route_priority")))
    if not route_priority:
        route_priority = tuple(profiles.keys())

    return FilterConfig(
        profiles=profiles,
        infer_fields=infer_fields,
        common_required_chunk_metadata=common_required,
        common_query_infer_fields=common_query_infer,
        route_priority=route_priority,
    )


@lru_cache(maxsize=4)
def _cached_filter_config(path_str: str) -> FilterConfig:
    return load_filter_config(Path(path_str))


def get_filter_config(path: Path | None = None) -> FilterConfig:
    config_path = path or Path(os.getenv("KB_FILTER_PROFILES_PATH", str(DEFAULT_PROFILES_PATH)))
    return _cached_filter_config(str(config_path.resolve()))


def reload_filter_config() -> None:
    _cached_filter_config.cache_clear()


def _profiles() -> dict[str, KnowledgeBaseProfile]:
    return get_filter_config().profiles


def _rules_by_key(rules: Iterable[FilterRule]) -> dict[str, FilterRule]:
    by_key: dict[str, FilterRule] = {}
    for rule in rules:
        by_key.setdefault(rule.filter_key, rule)
    return by_key


def all_filter_rules() -> tuple[FilterRule, ...]:
    return tuple(_rules_by_key(
        rule for profile in _profiles().values() for rule in profile.rules
    ).values())


FILTER_RULES: tuple[FilterRule, ...] = all_filter_rules()


def get_kb_profile(kb_id: str) -> KnowledgeBaseProfile | None:
    return _profiles().get(kb_id)


def rules_for_categories(categories: list[str] | None) -> tuple[FilterRule, ...]:
    profiles = _profiles()
    if not categories:
        return FILTER_RULES
    merged: dict[str, FilterRule] = {}
    known = False
    for category in categories:
        profile = profiles.get(category)
        if profile is None:
            continue
        known = True
        for rule in profile.rules:
            merged.setdefault(rule.filter_key, rule)
    return tuple(merged.values()) if known else FILTER_RULES


def supported_filter_keys(categories: list[str] | None = None) -> set[str]:
    return {rule.filter_key for rule in rules_for_categories(categories)}


def compact_filters(filters: MetadataFilters | None) -> MetadataFilters:
    return {k: v for k, v in (filters or {}).items() if v not in (None, "", [], ())}


def routable_kb_ids() -> tuple[str, ...]:
    """PDF KB ids for LLM routing (route_priority, excludes faq)."""
    config = get_filter_config()
    order = config.route_priority or tuple(config.profiles.keys())
    return tuple(kb_id for kb_id in order if kb_id in config.profiles and kb_id != "faq")


def query_infer_allowed_keys(knowledge_bases: list[str] | None) -> set[str]:
    """Fields allowed to be inferred from query — per-KB query_infer_fields only."""
    config = get_filter_config()
    allowed = set(config.common_query_infer_fields)
    if not knowledge_bases:
        return allowed
    for kb_id in knowledge_bases:
        profile = config.profiles.get(kb_id)
        if profile is not None:
            allowed.update(profile.query_infer_fields)
    return allowed


def _coerce_infer_value(spec: InferFieldSpec, match: re.Match[str]) -> Any:
    if spec.cast == "int":
        return int(match.group(1))
    if spec.cast == "upper":
        return match.group(0).upper()
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def infer_pdf_field_filters(
    query: str,
    *,
    knowledge_bases: list[str] | None = None,
) -> MetadataFilters:
    """Rule-extract fields listed in infer_fields + query_infer_fields contract."""
    filters: MetadataFilters = {}
    categories = [str(v) for v in (knowledge_bases or []) if v]
    allowed = query_infer_allowed_keys(categories or None)
    config = get_filter_config()

    for field_key, spec in config.infer_fields.items():
        if field_key not in allowed:
            continue
        match = spec.pattern.search(query)
        if match is None:
            continue
        filters[field_key] = _coerce_infer_value(spec, match)
    return filters


def infer_pdf_metadata_filters(
    query: str,
    *,
    knowledge_bases: list[str] | None = None,
) -> MetadataFilters:
    """Compose LLM/explicit category + rule-based field filters.

    ``knowledge_bases`` must be supplied by the caller (typically LLM router).
    Without it, only common query_infer_fields (e.g. doc_id) are applied.
    """
    filters: MetadataFilters = {}
    categories = [str(v) for v in (knowledge_bases or []) if v]
    if categories:
        filters["category"] = categories[0] if len(categories) == 1 else categories
    filters.update(infer_pdf_field_filters(query, knowledge_bases=categories or None))
    return compact_filters(filters)


def filter_categories(filters: MetadataFilters | None) -> list[str] | None:
    values = _as_list((filters or {}).get("category"))
    categories = [str(v) for v in values if v]
    return categories or None


def has_strict_filters(filters: MetadataFilters | None) -> bool:
    filters = compact_filters(filters)
    if not filters:
        return False
    rules = rules_for_categories(filter_categories(filters))
    return any(
        rule.mode is MatchMode.EXACT and _filter_value(filters, rule) is not None
        for rule in rules
    )


def merge_filters(
    base: MetadataFilters | None,
    override: MetadataFilters | None,
) -> MetadataFilters:
    merged = dict(base or {})
    merged.update(compact_filters(override))
    return compact_filters(merged)


def metadata_matches(metadata: dict[str, Any], filters: MetadataFilters | None) -> bool:
    filters = compact_filters(filters)
    if not filters:
        return True

    for rule in rules_for_categories(filter_categories(filters)):
        expected = _filter_value(filters, rule)
        if expected is None:
            continue
        if not _apply_rule(metadata, rule, expected):
            return False
    return True


def __getattr__(name: str) -> Any:
    if name == "KNOWLEDGE_BASE_PROFILES":
        return _profiles()
    raise AttributeError(name)
