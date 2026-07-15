"""KB schema gate and retrieval on_empty policy execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retrieval.core.filters import MetadataFilters, get_kb_profile


class SchemaGateError(Exception):
    def __init__(
        self,
        category: str,
        missing_fields: list[str],
        *,
        doc_id: str = "",
    ) -> None:
        self.category = category
        self.missing_fields = list(missing_fields)
        self.doc_id = doc_id
        fields = ", ".join(self.missing_fields)
        suffix = f" doc_id={doc_id}" if doc_id else ""
        super().__init__(f"schema gate failed for {category}{suffix}: missing {fields}")


@dataclass
class RetrievalTrace:
    query: str
    filters: MetadataFilters
    categories: list[str]
    vector_hits: int = 0
    lexical_hits: int = 0
    final_hits: int = 0
    on_empty_policy: str = "abstain"
    abstained: bool = False
    abstain_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def missing_required_fields(metadata: dict[str, Any], category: str) -> list[str]:
    profile = get_kb_profile(category)
    if profile is None:
        return []
    missing: list[str] = []
    for field_name in profile.required_chunk_metadata:
        value = metadata.get(field_name)
        if value in (None, "", [], ()):
            missing.append(field_name)
    return missing


def validate_chunk_metadata(
    metadata: dict[str, Any],
    category: str,
    *,
    doc_id: str = "",
) -> None:
    missing = missing_required_fields(metadata, category)
    if missing:
        raise SchemaGateError(category, missing, doc_id=doc_id or str(metadata.get("doc_id") or ""))


def validate_chunks_schema(chunks: list[Any], category: str) -> None:
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or chunk
        if not isinstance(metadata, dict):
            raise TypeError("chunk metadata must be a dict")
        validate_chunk_metadata(
            metadata,
            category,
            doc_id=str(metadata.get("doc_id") or ""),
        )


def resolve_on_empty_policy(categories: list[str] | None) -> str:
    if not categories:
        return "abstain"
    for category in categories:
        profile = get_kb_profile(category)
        if profile is not None:
            return profile.on_empty
    return "abstain"


def apply_on_empty_policy(
    hits: list[Any],
    *,
    query: str,
    filters: MetadataFilters | None,
    categories: list[str] | None,
    vector_hits: int,
    lexical_hits: int,
) -> tuple[list[Any], RetrievalTrace]:
    policy = resolve_on_empty_policy(categories)
    trace = RetrievalTrace(
        query=query,
        filters=dict(filters or {}),
        categories=list(categories or []),
        vector_hits=vector_hits,
        lexical_hits=lexical_hits,
        final_hits=len(hits),
        on_empty_policy=policy,
    )

    if hits:
        return hits, trace

    if policy == "abstain":
        trace.abstained = True
        trace.abstain_reason = "no_hits_after_metadata_filter"
        return [], trace

    return hits, trace
