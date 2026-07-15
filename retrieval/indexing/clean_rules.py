"""加载并应用 knowledge/raw/clean_rules.yaml 中的 PDF 清洗规则。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RULES_PATH = ROOT_DIR / "knowledge" / "raw" / "clean_rules.yaml"


@dataclass
class TableRules:
    skip_keywords: list[str] = field(default_factory=list)
    financial_keywords: list[str] = field(default_factory=list)
    default_class: str = "normal"


@dataclass
class CategoryRules:
    name: str
    chunk_strategy: str = "section"
    drop_block_types: set[str] = field(default_factory=set)
    keep_block_types: set[str] = field(default_factory=set)
    noise_paragraph_patterns: list[re.Pattern[str]] = field(default_factory=list)
    skip_section_keywords: list[str] = field(default_factory=list)
    skip_before_first_level1_title: bool = False
    table: TableRules = field(default_factory=TableRules)
    page_number_to_metadata: bool = True
    strip_checkbox_suffix: bool = True


@dataclass
class CleanRuleSet:
    global_rules: dict[str, Any]
    categories: dict[str, CategoryRules]

    def for_category(self, category: str) -> CategoryRules:
        return self.categories.get(category, self.categories["_default"])

    @classmethod
    def from_yaml(cls, path: Path = RULES_PATH) -> CleanRuleSet:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        global_cfg = raw.get("global", {})
        global_table = global_cfg.get("table", {})

        base_drop = set(global_cfg.get("drop_block_types", []))
        base_keep = set(global_cfg.get("keep_block_types", []))
        base_noise = _compile_patterns(global_cfg.get("noise_paragraph_patterns", []))
        base_table = TableRules(
            skip_keywords=list(global_table.get("skip_keywords", [])),
            financial_keywords=list(global_table.get("financial_keywords", [])),
            default_class=global_table.get("default_class", "normal"),
        )

        categories: dict[str, CategoryRules] = {}
        for name, cfg in raw.get("categories", {}).items():
            cover = cfg.get("cover", {})
            cat_table_cfg = {**global_table, **cfg.get("table", {})}
            cat_drop = (
                set(cfg["drop_block_types"])
                if "drop_block_types" in cfg
                else base_drop
            )
            cat_keep = base_keep | set(cfg.get("keep_block_types", []))
            categories[name] = CategoryRules(
                name=name,
                chunk_strategy=cfg.get("chunk_strategy", "section"),
                drop_block_types=cat_drop,
                keep_block_types=cat_keep,
                noise_paragraph_patterns=_compile_patterns(
                    list(global_cfg.get("noise_paragraph_patterns", []))
                    + list(cfg.get("noise_paragraph_patterns", []))
                ),
                skip_section_keywords=list(cfg.get("skip_section_keywords", [])),
                skip_before_first_level1_title=bool(
                    cover.get("skip_before_first_level1_title", False)
                ),
                table=TableRules(
                    skip_keywords=list(cat_table_cfg.get("skip_keywords", [])),
                    financial_keywords=list(cat_table_cfg.get("financial_keywords", [])),
                    default_class=cat_table_cfg.get("default_class", "normal"),
                ),
                page_number_to_metadata=bool(
                    global_cfg.get("page_number_to_metadata", True)
                ),
                strip_checkbox_suffix=bool(
                    global_cfg.get("strip_checkbox_suffix", True)
                ),
            )

        categories["_default"] = CategoryRules(
            name="_default",
            drop_block_types=base_drop,
            keep_block_types=base_keep,
            noise_paragraph_patterns=base_noise,
            table=base_table,
            page_number_to_metadata=bool(global_cfg.get("page_number_to_metadata", True)),
            strip_checkbox_suffix=bool(global_cfg.get("strip_checkbox_suffix", True)),
        )
        return cls(global_rules=global_cfg, categories=categories)


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns if p]


def _extract_text_from_block(block: dict[str, Any]) -> str:
    btype = block.get("type", "")
    content = block.get("content", {})

    if btype == "paragraph":
        parts = content.get("paragraph_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "title":
        parts = content.get("title_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "list":
        parts = content.get("list_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "page_header":
        parts = content.get("page_header_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "page_footer":
        parts = content.get("page_footer_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "page_number":
        parts = content.get("page_number_content", [])
        return "".join(p.get("content", "") for p in parts).strip()
    if btype == "table":
        return content.get("html", "") or ""
    if btype == "chart":
        return _extract_chart_text(block)
    if btype == "image":
        return _extract_image_text(block)
    return block.get("text", "").strip()


def _extract_image_text(block: dict[str, Any]) -> str:
    content = block.get("content", {})
    parts: list[str] = []
    for key in ("image_caption", "image_footnote"):
        for item in content.get(key) or []:
            text = item.get("content", "").strip()
            if text:
                parts.append(text)
    body = (content.get("content") or "").strip()
    if body:
        parts.append(body)
    sub_type = block.get("sub_type")
    if sub_type:
        parts.insert(0, f"[{sub_type}]")
    return "\n".join(parts).strip()


def _extract_chart_text(block: dict[str, Any]) -> str:
    content = block.get("content", {})
    parts: list[str] = []
    for key in ("chart_caption", "chart_footnote"):
        for item in content.get(key) or []:
            text = item.get("content", "").strip()
            if text:
                parts.append(text)
    body = (content.get("content") or "").strip()
    if body:
        parts.append(body)
    sub_type = block.get("sub_type")
    if sub_type:
        parts.insert(0, f"[{sub_type}]")
    return "\n".join(parts).strip()


def _title_level(block: dict[str, Any]) -> int | None:
    if block.get("type") != "title":
        return None
    return block.get("content", {}).get("level")


def parse_page_number_label(label: str) -> tuple[int | None, int | None]:
    """解析 '4 / 219' → (4, 219)。"""
    m = re.search(r"(\d+)\s*/\s*(\d+)", label)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def is_noise_paragraph(text: str, rules: CategoryRules) -> bool:
    t = text.strip()
    if not t:
        return True
    for pat in rules.noise_paragraph_patterns:
        if pat.search(t):
            return True
    return False


def clean_paragraph_text(text: str, rules: CategoryRules) -> str:
    if not rules.strip_checkbox_suffix:
        return text.strip()
    cleaned = re.sub(r"[□√]\s*(是|否|适用|不适用)\s*", "", text).strip()
    return cleaned


def should_skip_section(title: str, rules: CategoryRules) -> bool:
    for kw in rules.skip_section_keywords:
        if kw in title:
            return True
    return False


_FIRST_SENTENCE_RE = re.compile(r"^(.+?)(?:[。！？；：]|$)", re.DOTALL)
_FALLBACK_MAX_LEN = 80


def _first_sentence(text: str, max_len: int = _FALLBACK_MAX_LEN) -> str:
    """取正文第一句（至 。！？；： 或 max_len）。"""
    t = text.strip()
    if not t:
        return ""
    m = _FIRST_SENTENCE_RE.match(t)
    sentence = (m.group(1) if m else t).strip()
    if len(sentence) > max_len:
        return sentence[:max_len].rstrip()
    return sentence


def _chart_caption_from_text(text: str) -> str:
    for line in text.splitlines():
        ln = line.strip()
        if not ln or ln.startswith("[") or ln.startswith("|") or ln.startswith("```"):
            continue
        return ln
    return ""


def _table_caption_from_html(html: str) -> str:
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()
    return _first_sentence(plain)


def _fallback_section_path(
    block_type: str,
    text: str,
    *,
    caption: str = "",
) -> str:
    """无前置 title 时，从块内容推断章节名。"""
    if block_type == "chart":
        return caption.strip() or _chart_caption_from_text(text)
    if block_type == "table":
        return caption.strip() or _table_caption_from_html(text)
    if block_type in ("paragraph", "list"):
        return _first_sentence(text)
    return ""


def _resolve_section_path(
    section_path: str,
    block_type: str,
    text: str,
    *,
    caption: str = "",
) -> str:
    if section_path:
        return section_path
    return _fallback_section_path(block_type, text, caption=caption)


def classify_table(html: str, caption: str, section_path: str, rules: CategoryRules) -> str:
    blob = f"{caption} {section_path} {html}"
    for kw in rules.table.skip_keywords:
        if kw in blob:
            return "skip"
    for kw in rules.table.financial_keywords:
        if kw in blob:
            return "financial"
    return rules.table.default_class


def find_content_list_v2(parsed_dir: Path) -> Path | None:
    matches = sorted(parsed_dir.glob("*_content_list_v2.json"))
    return matches[0] if matches else None


def load_pages_from_v2(path: Path) -> list[list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"unexpected v2 format: {path}")
    return data


@dataclass
class CleanStats:
    dropped_by_type: dict[str, int]
    dropped_noise_paragraph: int
    dropped_cover_paragraph: int
    dropped_skip_section: int
    tables: dict[str, int]
    charts: int
    images: int
    kept_blocks: int

    def summary(self) -> str:
        lines = [
            f"kept_blocks={self.kept_blocks}",
            f"dropped_noise_paragraph={self.dropped_noise_paragraph}",
            f"dropped_cover_paragraph={self.dropped_cover_paragraph}",
            f"dropped_skip_section={self.dropped_skip_section}",
            f"dropped_by_type={self.dropped_by_type}",
            f"tables={self.tables}",
            f"charts={self.charts}",
            f"images={self.images}",
        ]
        return "\n".join(lines)


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._current_table = []
            return
        if tag == "tr" and self._current_table is not None:
            self._current_row = []
            return
        if tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._in_cell = True
            return
        if self._in_cell and tag in {"br", "p", "div"}:
            self._current_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None:
            text = _normalize_table_cell("".join(self._current_cell))
            self._current_row.append(text)
            self._current_cell = None
            self._in_cell = False
            return
        if tag == "tr" and self._current_table is not None and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
            return
        if tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)


def _normalize_table_cell(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def html_table_to_markdown(html_text: str) -> str:
    if "<table" not in html_text.lower():
        return html_text.strip()

    parser = _HtmlTableParser()
    parser.feed(html_text)
    markdown_tables: list[str] = []

    for table in parser.tables:
        if not table:
            continue
        width = max(len(row) for row in table)
        if width == 0:
            continue

        rows = [row + [""] * (width - len(row)) for row in table]
        header = rows[0]
        body = rows[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        markdown_tables.append("\n".join(lines))

    if markdown_tables:
        return "\n\n".join(markdown_tables).strip()
    return html_text.strip()


def _format_table_text(block: dict[str, Any], html: str) -> str:
    content = block.get("content", {})
    parts: list[str] = []
    for key in ("table_caption", "table_footnote"):
        for item in content.get(key) or []:
            if isinstance(item, dict):
                text = item.get("content", "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
    if html.strip():
        parts.append(html_table_to_markdown(html))
    return "\n".join(parts).strip()


@dataclass
class ExtractedBlock:
    block_type: str
    text: str
    section_path: str
    page_idx: int
    page_label: str | None = None
    table_class: str | None = None


@dataclass
class ExtractResult:
    blocks: list[ExtractedBlock]
    stats: CleanStats


def extract_blocks(
    pages: list[list[dict[str, Any]]],
    category: str,
    ruleset: CleanRuleSet | None = None,
    *,
    part_start: int = 1,
) -> ExtractResult:
    """按 clean_rules 过滤并返回保留的正文 block 列表。"""
    ruleset = ruleset or load_rules()
    rules = ruleset.for_category(category)

    dropped_by_type: dict[str, int] = {}
    tables: dict[str, int] = {"skip": 0, "normal": 0, "financial": 0}
    charts = 0
    images = 0
    dropped_noise = 0
    dropped_cover = 0
    dropped_section = 0
    kept = 0
    extracted: list[ExtractedBlock] = []

    seen_level1 = False
    skip_current_section = False
    section_path = ""

    for page_idx, page in enumerate(pages):
        page_label = ""
        for block in page:
            btype = block.get("type", "")

            if btype == "page_number":
                page_label = _extract_text_from_block(block)
                continue

            if btype in rules.drop_block_types:
                dropped_by_type[btype] = dropped_by_type.get(btype, 0) + 1
                continue

            if btype == "title":
                title = _extract_text_from_block(block)
                level = _title_level(block) or 1
                if level == 1:
                    seen_level1 = True
                skip_current_section = should_skip_section(title, rules)
                if skip_current_section:
                    dropped_section += 1
                    continue
                section_path = title
                kept += 1
                continue

            if skip_current_section:
                dropped_section += 1
                continue

            if btype == "paragraph":
                text = _extract_text_from_block(block)
                if rules.skip_before_first_level1_title and not seen_level1:
                    dropped_cover += 1
                    continue
                if is_noise_paragraph(text, rules):
                    dropped_noise += 1
                    continue
                cleaned = clean_paragraph_text(text, rules)
                if not cleaned:
                    dropped_noise += 1
                    continue
                extracted.append(
                    ExtractedBlock(
                        block_type="paragraph",
                        text=cleaned,
                        section_path=_resolve_section_path(
                            section_path, "paragraph", cleaned
                        ),
                        page_idx=page_idx,
                        page_label=page_label or None,
                    )
                )
                kept += 1
                continue

            if btype == "list":
                text = _extract_text_from_block(block)
                if not text:
                    continue
                extracted.append(
                    ExtractedBlock(
                        block_type="list",
                        text=text,
                        section_path=_resolve_section_path(section_path, "list", text),
                        page_idx=page_idx,
                        page_label=page_label or None,
                    )
                )
                kept += 1
                continue

            if btype == "table":
                html = _extract_text_from_block(block)
                caption = ""
                content = block.get("content", {})
                caps = content.get("table_caption") or []
                if caps:
                    caption = " ".join(
                        c.get("content", str(c)) if isinstance(c, dict) else str(c)
                        for c in caps
                    )
                tclass = classify_table(html, caption, section_path, rules)
                tables[tclass] = tables.get(tclass, 0) + 1
                if tclass == "skip":
                    continue
                text = _format_table_text(block, html)
                if not text:
                    continue
                extracted.append(
                    ExtractedBlock(
                        block_type="table",
                        text=text,
                        section_path=_resolve_section_path(
                            section_path, "table", text, caption=caption
                        ),
                        page_idx=page_idx,
                        page_label=page_label or None,
                        table_class=tclass,
                    )
                )
                kept += 1
                continue

            if btype == "chart":
                text = _extract_text_from_block(block)
                if not text:
                    continue
                charts += 1
                chart_caption = ""
                content = block.get("content", {})
                caps = content.get("chart_caption") or []
                if caps:
                    chart_caption = " ".join(
                        c.get("content", str(c)) if isinstance(c, dict) else str(c)
                        for c in caps
                    )
                extracted.append(
                    ExtractedBlock(
                        block_type="chart",
                        text=text,
                        section_path=_resolve_section_path(
                            section_path, "chart", text, caption=chart_caption
                        ),
                        page_idx=page_idx,
                        page_label=page_label or None,
                    )
                )
                kept += 1
                continue

            if btype == "image":
                text = _extract_text_from_block(block)
                if not text:
                    continue
                images += 1
                extracted.append(
                    ExtractedBlock(
                        block_type="image",
                        text=text,
                        section_path=_resolve_section_path(
                            section_path, "image", text
                        ),
                        page_idx=page_idx,
                        page_label=page_label or None,
                    )
                )
                kept += 1
                continue

            if btype in rules.keep_block_types:
                text = _extract_text_from_block(block)
                if text:
                    extracted.append(
                        ExtractedBlock(
                            block_type=btype,
                            text=text,
                            section_path=_resolve_section_path(
                                section_path, btype, text
                            ),
                            page_idx=page_idx,
                            page_label=page_label or None,
                        )
                    )
                    kept += 1

    stats = CleanStats(
        dropped_by_type=dropped_by_type,
        dropped_noise_paragraph=dropped_noise,
        dropped_cover_paragraph=dropped_cover,
        dropped_skip_section=dropped_section,
        tables=tables,
        charts=charts,
        images=images,
        kept_blocks=kept,
    )
    return ExtractResult(blocks=extracted, stats=stats)


def analyze_document(
    pages: list[list[dict[str, Any]]],
    category: str,
    ruleset: CleanRuleSet | None = None,
    *,
    part_start: int = 1,
) -> CleanStats:
    """dry-run：统计规则命中情况，不写库。"""
    return extract_blocks(
        pages, category, ruleset, part_start=part_start
    ).stats


@lru_cache(maxsize=1)
def load_rules() -> CleanRuleSet:
    return CleanRuleSet.from_yaml(RULES_PATH)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dry-run PDF clean rules stats")
    parser.add_argument("parsed_dir", type=Path, help="knowledge/parsed/... 解压目录")
    parser.add_argument(
        "--category",
        default="research_reports",
        help="manifest category，决定套用哪套规则",
    )
    args = parser.parse_args()

    v2 = find_content_list_v2(args.parsed_dir)
    if not v2:
        raise SystemExit(f"未找到 content_list_v2.json: {args.parsed_dir}")

    pages = load_pages_from_v2(v2)
    stats = analyze_document(pages, args.category)
    print(f"file: {v2}")
    print(f"category: {args.category}")
    print(stats.summary())


if __name__ == "__main__":
    main()
