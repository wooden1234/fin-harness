"""质量门的安全文本与缺口渲染。"""

from __future__ import annotations

from agents.orchestrator.contracts import Evidence

_RAW_JSON_MARKERS = ('"configured"', '"results"', "{'answer':", '"answer":')


def _escape_markdown_table_cell(value: str) -> str:
    """转义 Markdown 表格分隔符，并把换行限制在单元格内。"""
    return (
        str(value)
        .strip()
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def render_markdown_table(
    *,
    columns: list[str],
    rows: list[tuple[list[str], str]],
    title: str = "",
) -> str:
    """把已通过质量门的行渲染为 Markdown 表格。"""
    if not columns or not rows:
        return ""
    header = (
        "| "
        + " | ".join(_escape_markdown_table_cell(item) for item in columns)
        + " |"
    )
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rendered_rows: list[str] = []
    for cells, citation_markers in rows:
        rendered_cells = [_escape_markdown_table_cell(item) for item in cells]
        if citation_markers:
            rendered_cells[-1] += citation_markers
        rendered_rows.append("| " + " | ".join(rendered_cells) + " |")
    table = "\n".join([header, separator, *rendered_rows])
    safe_title = _escape_markdown_table_cell(title)
    return f"**{safe_title}**\n\n{table}" if safe_title else table


def _clean_display_text(item: Evidence) -> str:
    """只渲染明确标记可展示的文本，拒绝序列化工具响应。"""
    if not item.metadata.get("displayable"):
        return ""
    text = str(item.metadata.get("display_text") or item.content or "")
    text = " ".join(text.split())[:1000]
    if not text or any(marker in text for marker in _RAW_JSON_MARKERS):
        return ""
    if text.startswith(("{", "[")):
        return ""
    return text


clean_display_text = _clean_display_text

__all__ = ["clean_display_text", "render_markdown_table"]
