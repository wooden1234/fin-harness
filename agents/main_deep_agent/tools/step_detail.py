"""面向用户的工具步骤摘要：只暴露可展示字段，不回传原始工具 JSON。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agents.orchestrator.contracts import Evidence

_MAX_QUERY_CHARS = 200
_MAX_DISPLAY_CHARS = 800
_MAX_ROWS = 8
_MAX_CELL_CHARS = 80

_FAMILY_TITLES = {
    "weather": "天气数据",
    "market": "问财数据",
    "web": "联网资料",
    "research": "研报与公告",
    "financial": "财务数据",
    "knowledge": "知识库",
    "calculation": "受限计算",
}


def _cell(value: object) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > _MAX_CELL_CHARS:
        return text[: _MAX_CELL_CHARS - 1] + "…"
    return text or "—"


def _format_fact_value(fact: Mapping[str, Any]) -> str:
    raw = fact.get("value")
    unit = str(fact.get("unit") or "").strip()
    currency = str(fact.get("currency") or "").strip()
    if raw is None or raw == "":
        text = str(fact.get("text") or "").strip()
        return _cell(text) if text else "—"
    try:
        number = float(raw)
        if abs(number) >= 100_000_000:
            rendered = f"{number / 100_000_000:.2f}亿"
        elif abs(number) >= 10_000:
            rendered = f"{number / 10_000:.2f}万"
        elif abs(number - round(number)) < 1e-9:
            rendered = str(int(round(number)))
        else:
            rendered = f"{number:.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        rendered = str(raw)
    suffix = unit or currency
    return _cell(f"{rendered}{suffix}" if suffix else rendered)


def _rows_from_facts(evidence: Sequence[Evidence]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in evidence:
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        for fact in list(metadata.get("facts") or []):
            if not isinstance(fact, Mapping):
                continue
            rows.append(
                [
                    _cell(fact.get("entity") or metadata.get("entity")),
                    _cell(fact.get("metric")),
                    _cell(fact.get("fiscal_period") or metadata.get("fiscal_period")),
                    _format_fact_value(fact),
                ]
            )
            if len(rows) >= _MAX_ROWS:
                return rows
    return rows


def _table_from_preview(evidence: Sequence[Evidence]) -> dict[str, Any] | None:
    for item in evidence:
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        preview = metadata.get("preview_table")
        if not isinstance(preview, Mapping):
            continue
        columns = preview.get("columns")
        rows = preview.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        clean_columns = [_cell(column) for column in columns[:8] if str(column).strip()]
        clean_rows: list[list[str]] = []
        for row in rows[:_MAX_ROWS]:
            if not isinstance(row, (list, tuple)):
                continue
            cells = [_cell(cell) for cell in list(row)[: len(clean_columns)]]
            if len(cells) == len(clean_columns):
                clean_rows.append(cells)
        if clean_columns and clean_rows:
            result: dict[str, Any] = {
                "columns": clean_columns,
                "rows": clean_rows,
            }
            query = str(preview.get("query") or "").strip()[:_MAX_QUERY_CHARS]
            if query:
                result["query"] = query
            return result
    return None


def _select_displayable(evidence: Sequence[Evidence]) -> list[Evidence]:
    selected = [item for item in evidence if item.metadata.get("displayable")]
    if selected:
        return selected[:3]
    if evidence:
        return [evidence[0]]
    return []


def build_public_tool_step_detail(
    *,
    tool_id: str,
    source_family: str,
    query: str = "",
    entity: str = "",
    evidence: Sequence[Evidence] | None = None,
    status: str = "done",
    error: str = "",
) -> dict[str, Any] | None:
    """构造可下发给前端的步骤 detail；无可展示内容时返回 None。"""
    detail: dict[str, Any] = {
        "title": _FAMILY_TITLES.get(source_family, "资料查询"),
    }
    cleaned_query = " ".join(str(query or "").split()).strip()[:_MAX_QUERY_CHARS]
    if cleaned_query:
        detail["query"] = cleaned_query
    if entity and not cleaned_query:
        detail["query"] = str(entity).strip()[:_MAX_QUERY_CHARS]

    if status == "running":
        return detail if "query" in detail else None

    selected = _select_displayable(list(evidence or []))
    preview = _table_from_preview(selected)
    if preview is not None:
        if preview.get("query") and "query" not in detail:
            detail["query"] = preview["query"]
        detail["columns"] = preview["columns"]
        detail["rows"] = preview["rows"]

    texts: list[str] = []
    for item in selected:
        metadata = item.metadata if isinstance(item.metadata, Mapping) else {}
        text = str(metadata.get("display_text") or item.title or "").strip()
        text = " ".join(text.split())
        # 有结构化预览表时跳过 key=value 长串，只用短摘要。
        if preview is not None and ("=" in text and "，" in text):
            continue
        if text and not text.startswith(("{", "[")):
            texts.append(text[:400])
    if texts:
        detail["display_text"] = "；".join(texts)[:_MAX_DISPLAY_CHARS]
    elif preview is not None and selected:
        # 短摘要优先；否则不塞长串
        metadata = selected[0].metadata if isinstance(selected[0].metadata, Mapping) else {}
        short = str(metadata.get("display_text") or "").strip()
        if short and not (short.startswith("问财「") and "=" in short):
            detail["display_text"] = short[:_MAX_DISPLAY_CHARS]

    if "columns" not in detail:
        rows = _rows_from_facts(selected)
        if rows:
            detail["columns"] = ["实体", "指标", "期间", "数值"]
            detail["rows"] = rows

    if status == "error" and "display_text" not in detail and "rows" not in detail:
        if error:
            detail["error"] = error[:120]
        return detail if "query" in detail else None

    if "display_text" not in detail and "rows" not in detail and "query" not in detail:
        return None
    detail["tool_id"] = tool_id
    return detail


__all__ = ["build_public_tool_step_detail"]
