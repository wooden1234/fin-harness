"""PDF 清洗切块：manifest_pdf.yaml + clean_rules.yaml → chunks / TextNode。"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR_STR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_BACKEND_DIR_STR = os.path.abspath(os.path.join(_ROOT_DIR_STR, "app", "backend"))
if sys.path and os.path.abspath(sys.path[0]) == _SCRIPT_DIR:
    sys.path.pop(0)
for _path in (_BACKEND_DIR_STR, _ROOT_DIR_STR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(_ROOT_DIR_STR) / ".env")

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from retrieval.pdf_processing.clean_rules import (
    CleanRuleSet,
    ExtractedBlock,
    extract_blocks,
    find_content_list_v2,
    load_pages_from_v2,
    load_rules,
    parse_page_number_label,
)

ROOT_DIR = Path(_ROOT_DIR_STR)



MANIFEST_PATH = ROOT_DIR / "knowledge" / "raw" / "manifest_pdf.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "knowledge" / "cleaned"
INGEST_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "ingest_manifest.json"

logger = logging.getLogger(__name__)

NARRATIVE_TYPES = frozenset({"paragraph", "list"})
ATOMIC_TYPES = frozenset({"table", "chart", "image"})
_SPLIT_PAGE_RE = re.compile(r"_p(?P<start>\d{3})-(?P<end>\d{3})$")


@dataclass
class ParsedPart:
    parsed_path: Path
    page_range: str | None = None
    part_index: int = 1
    part_start: int = 1


@dataclass
class IngestJob:
    doc_id: str
    title: str
    category: str
    file: str
    chunk_strategy: str
    authority_tier: str
    parts: list[ParsedPart] = field(default_factory=list)
    ticker: str | None = None
    fiscal_year: int | None = None
    effective_date: str | None = None
    issuer: str | None = None
    language: str | None = None
    doc_group: str | None = None


@dataclass
class ChunkRecord:
    text: str
    metadata: dict[str, Any]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_page_range(page_range: str | None) -> int:
    if not page_range:
        return 1
    m = re.match(r"(\d+)", page_range)
    return int(m.group(1)) if m else 1


def _part_start_from_dir(parsed_dir: Path) -> int:
    m = _SPLIT_PAGE_RE.search(parsed_dir.name)
    return int(m.group("start")) if m else 1


def discover_ingest_jobs(
    manifest: dict[str, Any],
    *,
    categories: Iterable[str] | None = None,
    doc_ids: Iterable[str] | None = None,
) -> list[IngestJob]:
    allowed_cats = set(categories) if categories else None
    allowed_docs = set(doc_ids) if doc_ids else None
    jobs: list[IngestJob] = []

    for category, docs in manifest.get("documents", {}).items():
        if allowed_cats and category not in allowed_cats:
            continue
        cat_cfg = manifest.get("categories", {}).get(category, {})
        default_strategy = cat_cfg.get("chunk_strategy", "section")

        for doc in docs:
            if not doc.get("ingest", False):
                continue
            doc_id = doc["doc_id"]
            if allowed_docs and doc_id not in allowed_docs:
                continue

            parts: list[ParsedPart] = []
            if doc.get("parts"):
                for part in doc["parts"]:
                    parsed_path = ROOT_DIR / part["parsed_path"]
                    page_range = part.get("page_range")
                    parts.append(
                        ParsedPart(
                            parsed_path=parsed_path,
                            page_range=page_range,
                            part_index=int(part.get("part_index", len(parts) + 1)),
                            part_start=_parse_page_range(page_range),
                        )
                    )
            elif doc.get("parsed_path"):
                parsed_path = ROOT_DIR / doc["parsed_path"]
                parts.append(
                    ParsedPart(
                        parsed_path=parsed_path,
                        part_index=1,
                        part_start=_part_start_from_dir(parsed_path),
                    )
                )

            if not parts:
                logger.warning(f"跳过无 parsed_path 的文档: {doc_id}")
                continue

            jobs.append(
                IngestJob(
                    doc_id=doc_id,
                    title=doc["title"],
                    category=category,
                    file=doc["file"],
                    chunk_strategy=doc.get("chunk_strategy", default_strategy),
                    authority_tier=doc.get("authority_tier", "official"),
                    parts=parts,
                    ticker=doc.get("ticker"),
                    fiscal_year=doc.get("fiscal_year"),
                    effective_date=doc.get("effective_date"),
                    issuer=doc.get("issuer"),
                    language=doc.get("language"),
                    doc_group=doc.get("doc_group") or Path(doc["file"]).stem,
                )
            )
    return jobs


def resolve_page_num(
    block: ExtractedBlock,
    part_start: int,
    page_cfg: dict[str, Any],
) -> int:
    if page_cfg.get("use_printed_page_number") and block.page_label:
        printed, _ = parse_page_number_label(block.page_label)
        if printed is not None:
            return printed
    if page_cfg.get("use_page_idx", True):
        return part_start + block.page_idx
    return block.page_idx + 1


def _chunk_prefix(
    block_type: str,
    *,
    title: str,
    section_path: str,
    page_num: int,
    prefix_cfg: dict[str, str],
    table_class: str | None = None,
) -> str:
    if block_type == "table":
        if table_class == "financial":
            template = prefix_cfg.get("table", "")
        else:
            template = prefix_cfg.get("table", "")
    elif block_type == "chart":
        template = prefix_cfg.get("chart", "")
    elif block_type == "image":
        template = prefix_cfg.get("image") or prefix_cfg.get("chart", "")
    else:
        template = prefix_cfg.get("narrative", "")

    if not template:
        return ""
    return template.format(
        title=title,
        ticker="",
        fiscal_year="",
        section_path=section_path or "未分类",
        page_num=page_num,
    )


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _markdown_row_cells(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("|") or "|" not in line[1:]:
        return None
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line[1:].split("|")]


def _is_markdown_separator(line: str) -> bool:
    cells = _markdown_row_cells(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _table_width(line: str) -> int:
    cells = _markdown_row_cells(line)
    return len(cells or [])


def _split_markdown_table_text(
    text: str,
) -> tuple[list[str], list[str] | None, list[str], bool]:
    """Return preamble, header lines, body row lines, and whether text had a header."""

    preamble: list[str] = []
    table_lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        if _markdown_row_cells(line) is not None:
            in_table = True
            table_lines.append(line.rstrip())
        elif not in_table:
            preamble.append(line.rstrip())
        elif line.strip():
            # Keep trailing notes as preamble-like context for all split pieces.
            preamble.append(line.rstrip())

    if not table_lines:
        return preamble, None, [], False

    if len(table_lines) >= 2 and _is_markdown_separator(table_lines[1]):
        return preamble, table_lines[:2], table_lines[2:], True
    return preamble, None, table_lines, False


def _split_table_text_by_rows(
    text: str,
    max_chars: int,
    inherited_header: list[str] | None = None,
) -> tuple[list[str], list[str] | None, bool]:
    """Split a markdown table by rows while preserving table headers."""

    preamble, detected_header, body_rows, had_header = _split_markdown_table_text(text)
    header = detected_header or inherited_header
    if detected_header:
        inherited_header = detected_header

    if not body_rows or not header:
        return _split_long_text(text, max_chars, 0), inherited_header, had_header

    header_width = _table_width(header[0])
    compatible_rows = [
        row for row in body_rows if _table_width(row) == header_width or _table_width(row) == 0
    ]
    if len(compatible_rows) != len(body_rows) and not detected_header:
        return _split_long_text(text, max_chars, 0), inherited_header, had_header

    prefix_lines = [line for line in preamble if line.strip()]
    fixed_lines = prefix_lines + header
    fixed_text = "\n".join(fixed_lines)
    pieces: list[str] = []
    current_rows: list[str] = []

    def flush_rows() -> None:
        nonlocal current_rows
        if not current_rows:
            return
        pieces.append("\n".join(fixed_lines + current_rows).strip())
        current_rows = []

    for row in body_rows:
        candidate = "\n".join(fixed_lines + current_rows + [row]).strip()
        if current_rows and len(candidate) > max_chars:
            flush_rows()
        current_rows.append(row)

    flush_rows()
    if not pieces and fixed_text:
        pieces.append(fixed_text)
    return pieces, inherited_header, had_header


def _flush_narrative_buffer(
    buffer: list[str],
    *,
    job: IngestJob,
    section_path: str,
    page_num: int,
    chunk_cfg: dict[str, Any],
    prefix_cfg: dict[str, str],
    part: ParsedPart,
    chunk_index: int,
) -> tuple[list[ChunkRecord], int]:
    if not buffer:
        return [], chunk_index

    max_chars = int(chunk_cfg.get("max_chars", 512))
    overlap = int(chunk_cfg.get("overlap", 64))
    embed_prefix = bool(chunk_cfg.get("embed_section_in_text", True))
    body = "\n\n".join(buffer)
    chunks: list[ChunkRecord] = []

    for piece in _split_long_text(body, max_chars, overlap):
        prefix = ""
        if embed_prefix:
            prefix = _chunk_prefix(
                "paragraph",
                title=job.title,
                section_path=section_path,
                page_num=page_num,
                prefix_cfg=prefix_cfg,
            )
        text = f"{prefix}\n{piece}".strip() if prefix else piece
        chunks.append(
            ChunkRecord(
                text=text,
                metadata=_base_chunk_metadata(
                    job,
                    part,
                    chunk_index=chunk_index,
                    section_path=section_path,
                    page_num=page_num,
                    block_type="narrative",
                ),
            )
        )
        chunk_index += 1
    return chunks, chunk_index


def _base_chunk_metadata(
    job: IngestJob,
    part: ParsedPart,
    *,
    chunk_index: int,
    section_path: str,
    page_num: int,
    block_type: str,
    table_class: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "format": "pdf",
        "doc_id": job.doc_id,
        "title": job.title,
        "category": job.category,
        "source": Path(job.file).name,
        "file": job.file,
        "authority_tier": job.authority_tier,
        "chunk_strategy": job.chunk_strategy,
        "section_path": section_path or "未分类",
        "section": section_path or "未分类",
        "page_num": page_num,
        "part_index": part.part_index,
        "chunk_index": chunk_index,
        "block_type": block_type,
        "doc_type": job.category,
    }
    if part.page_range:
        meta["page_range"] = part.page_range
    if job.ticker:
        meta["ticker"] = job.ticker
    if job.fiscal_year is not None:
        meta["fiscal_year"] = job.fiscal_year
    if job.effective_date:
        meta["effective_date"] = job.effective_date
    if job.issuer:
        meta["issuer"] = job.issuer
    if job.language:
        meta["language"] = job.language
    if job.doc_group:
        meta["doc_group"] = job.doc_group
    if table_class:
        meta["table_class"] = table_class
    if extra:
        meta.update(extra)
    return meta


def blocks_to_markdown(
    blocks: list[ExtractedBlock],
    job: IngestJob,
    ruleset: CleanRuleSet,
    *,
    part: ParsedPart | None = None,
    include_header: bool = True,
) -> str:
    """将 extract_blocks 结果渲染为可读 Markdown（无 chunk 前缀、不切块）。"""
    page_cfg = ruleset.global_rules.get("page", {})
    part_start = part.part_start if part else 1
    lines: list[str] = []

    if include_header:
        lines.extend([f"# {job.title}", ""])
        if job.issuer:
            lines.extend([f"- **issuer**: {job.issuer}", ""])
        if job.effective_date:
            lines.extend([f"- **date**: {job.effective_date}", ""])
        lines.append("")

    last_section: str | None = None

    for block in blocks:
        section = block.section_path or "未分类"
        if section != last_section:
            lines.extend([f"## {section}", ""])
            last_section = section

        page_num = resolve_page_num(block, part_start, page_cfg)
        page_hint = f"<!-- p.{page_num} -->"

        if block.block_type == "table":
            lines.extend([page_hint, block.text, ""])
        elif block.block_type in ("chart", "image"):
            label = "图表" if block.block_type == "chart" else "图片"
            lines.extend([page_hint, f"<!-- {label} -->", ""])
            body = block.text.strip()
            if body.startswith("```"):
                lines.extend([body, ""])
            else:
                lines.extend([body, ""])
        else:
            lines.extend([block.text, ""])

    return "\n".join(lines).strip()


def write_cleaned_md(
    job: IngestJob,
    parts_blocks: list[tuple[ParsedPart, list[ExtractedBlock]]],
    ruleset: CleanRuleSet,
    output_dir: Path,
) -> Path:
    out_dir = output_dir / job.category / job.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cleaned.md"

    sections: list[str] = []
    for idx, (part, blocks) in enumerate(parts_blocks):
        if len(job.parts) > 1:
            label = part.page_range or f"part {part.part_index}"
            sections.append(f"<!-- part {part.part_index}: {label} -->\n")
        sections.append(
            blocks_to_markdown(
                blocks,
                job,
                ruleset,
                part=part,
                include_header=(idx == 0),
            )
        )

    out_path.write_text("\n\n".join(s for s in sections if s).strip() + "\n", encoding="utf-8")
    return out_path


def chunk_blocks(
    blocks: list[ExtractedBlock],
    job: IngestJob,
    part: ParsedPart,
    ruleset: CleanRuleSet,
) -> list[ChunkRecord]:
    global_cfg = ruleset.global_rules
    prefix_cfg = global_cfg.get("chunk_prefix", {})
    page_cfg = global_cfg.get("page", {})
    strategy = job.chunk_strategy
    category_rules = ruleset.for_category(job.category)
    chunk_cfg = category_rules.chunk or global_cfg.get("chunk", {})

    chunks: list[ChunkRecord] = []
    chunk_index = 0
    narrative_buffer: list[str] = []
    current_section = ""
    current_page = part.part_start
    table_headers: dict[tuple[str, str], list[str]] = {}

    def flush_buffer() -> None:
        nonlocal chunk_index, narrative_buffer, current_section, current_page
        new_chunks, chunk_index = _flush_narrative_buffer(
            narrative_buffer,
            job=job,
            section_path=current_section,
            page_num=current_page,
            chunk_cfg=chunk_cfg,
            prefix_cfg=prefix_cfg,
            part=part,
            chunk_index=chunk_index,
        )
        chunks.extend(new_chunks)
        narrative_buffer = []

    for block in blocks:
        page_num = resolve_page_num(block, part.part_start, page_cfg)
        if block.section_path != current_section and narrative_buffer:
            flush_buffer()
        current_section = block.section_path
        current_page = page_num

        if block.block_type in NARRATIVE_TYPES:
            if strategy == "table_aware" and narrative_buffer:
                flush_buffer()
            narrative_buffer.append(block.text)
            if sum(len(x) for x in narrative_buffer) + 2 * (len(narrative_buffer) - 1) >= int(
                chunk_cfg.get("max_chars", 512)
            ):
                flush_buffer()
            continue

        flush_buffer()

        if block.block_type in ATOMIC_TYPES:
            prefix = _chunk_prefix(
                block.block_type,
                title=job.title,
                section_path=block.section_path,
                page_num=page_num,
                prefix_cfg=prefix_cfg,
                table_class=block.table_class,
            )
            max_chars = int(chunk_cfg.get("max_chars", 512))
            if block.block_type == "table":
                header_key = (block.section_path or "", block.table_class or "")
                inherited_header = table_headers.get(header_key)
                pieces, updated_header, had_header = _split_table_text_by_rows(
                    block.text,
                    max_chars,
                    inherited_header=inherited_header,
                )
                if updated_header:
                    table_headers[header_key] = updated_header
                table_extra = {
                    "table_split_strategy": (
                        "financial_row_aware"
                        if block.table_class == "financial"
                        else "table_row_aware"
                    ),
                    "table_header_inherited": bool(inherited_header and not had_header),
                }
            else:
                pieces = _split_long_text(
                    block.text,
                    max_chars,
                    int(chunk_cfg.get("overlap", 64)),
                )
                table_extra = {}

            part_count = len(pieces)
            for part_idx, piece in enumerate(pieces):
                text = f"{prefix}\n{piece}".strip() if prefix else piece
                extra = dict(table_extra)
                if block.block_type == "table":
                    extra.update(
                        {
                            "table_part_index": part_idx,
                            "table_part_count": part_count,
                        }
                    )
                chunks.append(
                    ChunkRecord(
                        text=text,
                        metadata=_base_chunk_metadata(
                            job,
                            part,
                            chunk_index=chunk_index,
                            section_path=block.section_path,
                            page_num=page_num,
                            block_type=block.block_type,
                            table_class=block.table_class,
                            extra=extra,
                        ),
                    )
                )
                chunk_index += 1
            continue

        narrative_buffer.append(block.text)

    flush_buffer()
    return chunks


def process_part(
    job: IngestJob,
    part: ParsedPart,
    ruleset: CleanRuleSet,
    *,
    make_chunks: bool = True,
) -> tuple[list[ChunkRecord], list[ExtractedBlock], dict[str, Any]]:
    v2 = find_content_list_v2(part.parsed_path)
    if not v2:
        raise FileNotFoundError(f"未找到 content_list_v2.json: {part.parsed_path}")

    pages = load_pages_from_v2(v2)
    result = extract_blocks(
        pages,
        job.category,
        ruleset,
        part_start=part.part_start,
    )
    chunks = chunk_blocks(result.blocks, job, part, ruleset) if make_chunks else []
    part_summary = {
        "parsed_path": str(part.parsed_path.relative_to(ROOT_DIR)),
        "page_range": part.page_range,
        "part_index": part.part_index,
        "kept_blocks": result.stats.kept_blocks,
        "chunks": len(chunks),
        "stats": asdict(result.stats),
    }
    return chunks, result.blocks, part_summary


def process_job(
    job: IngestJob,
    ruleset: CleanRuleSet,
    *,
    make_chunks: bool = True,
) -> tuple[list[ChunkRecord], list[tuple[ParsedPart, list[ExtractedBlock]]], dict[str, Any]]:
    all_chunks: list[ChunkRecord] = []
    parts_blocks: list[tuple[ParsedPart, list[ExtractedBlock]]] = []
    part_summaries: list[dict[str, Any]] = []

    for part in job.parts:
        if not part.parsed_path.is_dir():
            raise FileNotFoundError(f"解压目录不存在: {part.parsed_path}")
        part_chunks, blocks, summary = process_part(job, part, ruleset, make_chunks=make_chunks)
        parts_blocks.append((part, blocks))
        offset = len(all_chunks)
        for i, chunk in enumerate(part_chunks):
            chunk.metadata["chunk_index"] = offset + i
        all_chunks.extend(part_chunks)
        part_summaries.append(summary)

    job_summary = {
        "doc_id": job.doc_id,
        "title": job.title,
        "category": job.category,
        "file": job.file,
        "chunk_strategy": job.chunk_strategy,
        "parts": part_summaries,
        "chunks": len(all_chunks),
        "kept_blocks": sum(s.get("kept_blocks", 0) for s in part_summaries),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return all_chunks, parts_blocks, job_summary


def write_job_output(
    job: IngestJob,
    chunks: list[ChunkRecord],
    output_dir: Path,
) -> Path:
    out_dir = output_dir / job.category / job.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk.text, "metadata": chunk.metadata}, ensure_ascii=False))
            f.write("\n")
    return out_path


def chunks_to_nodes(chunks: list[ChunkRecord]) -> list[Any]:
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    nodes: list[Any] = []
    for chunk in chunks:
        # LlamaIndex 写入 PG 时会用 ref_doc_id 覆盖扁平 doc_id，必须设 SOURCE。
        doc_id = str(chunk.metadata.get("doc_id") or "").strip()
        chunk_index = int(chunk.metadata.get("chunk_index") or 0)
        chunk_id = str(chunk.metadata.get("chunk_id") or f"{doc_id}:L3:{chunk_index:06d}")
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata.setdefault("chunk_level", "L3")
        relationships = {}
        if doc_id and doc_id != "None":
            relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
        nodes.append(
            TextNode(
                id_=chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                relationships=relationships,
            )
        )
    return nodes


def _load_ingest_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"documents": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_ingest_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_output_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _successful_doc_ids(report: dict[str, Any]) -> set[str]:
    doc_ids: set[str] = set()
    for doc_id, summary in report.get("documents", {}).items():
        if summary.get("status") == "schema_rejected":
            continue
        if summary.get("chunks", 0) or summary.get("output"):
            doc_ids.add(str(doc_id))
    return doc_ids


def build_parent_outputs(output_dir: Path, doc_ids: set[str] | None = None) -> dict[str, int]:
    """为本次处理的文档生成 parent_chunks.jsonl。"""
    from retrieval.indexing.scripts.build_parent_chunks import write_parent_chunks

    counts: dict[str, int] = {}
    for chunks_path in sorted(output_dir.glob("**/chunks.jsonl")):
        doc_id = chunks_path.parent.name
        if doc_ids and doc_id not in doc_ids:
            continue
        _, count = write_parent_chunks(chunks_path)
        counts[doc_id] = count
    return counts


def run_ingest_pdf(
    *,
    categories: Iterable[str] | None = None,
    doc_ids: Iterable[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_output: bool = True,
    export_md: bool = True,
    export_chunks: bool = True,
    manifest_path: Path = MANIFEST_PATH,
    skip_schema_gate: bool = False,
) -> tuple[list[ChunkRecord], dict[str, Any]]:
    output_dir = _normalize_output_dir(output_dir)
    manifest = load_manifest(manifest_path)
    ruleset = load_rules()
    jobs = discover_ingest_jobs(manifest, categories=categories, doc_ids=doc_ids)

    if not jobs:
        logger.warning("未找到可 ingest 的 PDF 文档")
        return [], {"documents": {}, "total_chunks": 0}

    ingest_manifest_path = output_dir / "ingest_manifest.json"
    ingest_manifest = _load_ingest_manifest(ingest_manifest_path)
    all_chunks: list[ChunkRecord] = []
    summaries: dict[str, Any] = {}

    from retrieval.core.kb_contract import SchemaGateError, validate_chunks_schema

    for job in jobs:
        logger.info(f"ingest {job.doc_id} ({job.category}) parts={len(job.parts)}")
        try:
            chunks, parts_blocks, summary = process_job(
                job, ruleset, make_chunks=export_chunks
            )
            if export_chunks and chunks and not skip_schema_gate:
                validate_chunks_schema(chunks, job.category)
        except SchemaGateError as exc:
            logger.error("schema gate rejected doc_id=%s category=%s: %s", job.doc_id, job.category, exc)
            summaries[job.doc_id] = {
                "doc_id": job.doc_id,
                "category": job.category,
                "status": "schema_rejected",
                "error": str(exc),
                "missing_fields": exc.missing_fields,
            }
            continue
        if write_output:
            if export_chunks:
                out_path = write_job_output(job, chunks, output_dir)
                summary["output"] = _display_path(out_path)
            if export_md:
                md_path = write_cleaned_md(job, parts_blocks, ruleset, output_dir)
                summary["cleaned_md"] = _display_path(md_path)
        summaries[job.doc_id] = summary
        if export_chunks:
            all_chunks.extend(chunks)
        logger.info(
            f"  → kept_blocks={summary.get('kept_blocks', 0)}"
            + (f" chunks={len(chunks)}" if export_chunks else " (md-only)")
        )

    report = {
        "documents": summaries,
        "total_documents": len(summaries),
        "total_chunks": sum(s.get("chunks", 0) for s in summaries.values()),
    }
    if write_output:
        ingest_manifest["documents"].update(summaries)
        ingest_manifest["total_chunks"] = sum(
            s.get("chunks", 0) for s in ingest_manifest["documents"].values()
        )
        _save_ingest_manifest(ingest_manifest_path, ingest_manifest)

    return all_chunks, report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="PDF 清洗切块：manifest + clean_rules → knowledge/cleaned"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="仅处理指定 category，如 research_reports annual_reports",
    )
    parser.add_argument(
        "--doc-id",
        nargs="+",
        default=None,
        help="仅处理指定 doc_id，如 PDF-RR-20260330",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="chunk 输出根目录，默认 knowledge/cleaned",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="不写 chunks.jsonl / cleaned.md，仅统计",
    )
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--no-export-md",
        action="store_true",
        help="不写 cleaned.md（默认与 chunks.jsonl 一并输出）",
    )
    export_group.add_argument(
        "--md-only",
        action="store_true",
        help="只写 cleaned.md，不写 chunks.jsonl",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="旧链路：切块完成后写入 pgvector，并在配置了 ELASTICSEARCH_URL 时双写 ES BM25"
        "（需 Embedding；与 --md-only 不兼容）",
    )
    parser.add_argument(
        "--build-parent-chunks",
        action="store_true",
        help="根据 L3 chunks 生成 L1/L2 parent_chunks.jsonl",
    )
    parser.add_argument(
        "--write-parent-pg",
        action="store_true",
        help="将 documents、parent_chunks、chunk_registry 写入 PostgreSQL rag schema",
    )
    parser.add_argument(
        "--index-es-leaf",
        action="store_true",
        help="将 L3 chunks 写入 Elasticsearch BM25 索引",
    )
    parser.add_argument(
        "--index-milvus-leaf",
        action="store_true",
        help="将 L3 chunks 写入 Milvus 向量索引",
    )
    parser.add_argument(
        "--rebuild-stores",
        action="store_true",
        help="重建本次处理文档对应的 PG/ES/Milvus 数据；仅对显式写入的 store 生效",
    )
    parser.add_argument(
        "--skip-schema-gate",
        action="store_true",
        help="跳过 KB schema 门禁（仅调试；缺 metadata 的 chunk 不应入库）",
    )
    args = parser.parse_args()

    if args.md_only and args.rebuild_index:
        parser.error("--md-only 与 --rebuild-index 不能同时使用")
    if args.md_only and (
        args.build_parent_chunks
        or args.write_parent_pg
        or args.index_es_leaf
        or args.index_milvus_leaf
    ):
        parser.error("--md-only 不能与 parent/PG/ES/Milvus 入库同时使用")
    if args.no_write and (args.build_parent_chunks or args.write_parent_pg):
        parser.error("--no-write 下无法生成或写入 parent_chunks.jsonl")

    export_md = not args.no_export_md
    export_chunks = not args.md_only
    output_dir = _normalize_output_dir(args.output_dir)

    chunks, report = run_ingest_pdf(
        categories=args.categories,
        doc_ids=args.doc_id,
        output_dir=output_dir,
        write_output=not args.no_write,
        export_md=export_md,
        export_chunks=export_chunks,
        skip_schema_gate=args.skip_schema_gate,
    )

    print(f"documents={report['total_documents']} chunks={report['total_chunks']}")
    for doc_id, summary in report.get("documents", {}).items():
        parts = []
        if summary.get("output"):
            parts.append(f"chunks={summary.get('chunks', 0)}")
        if summary.get("cleaned_md"):
            parts.append(f"md={summary['cleaned_md']}")
        print(f"  {doc_id}: " + " ".join(parts) if parts else f"  {doc_id}: (stats only)")

    doc_ids = _successful_doc_ids(report)
    needs_parent = args.build_parent_chunks or args.write_parent_pg or args.index_es_leaf or args.index_milvus_leaf
    if needs_parent and doc_ids:
        parent_counts = build_parent_outputs(output_dir, doc_ids=doc_ids)
        total_parents = sum(parent_counts.values())
        print(f"parent chunks built: documents={len(parent_counts)} parent_chunks={total_parents}")

        from retrieval.indexing.scripts.build_parent_chunks import (
            enrich_leaf_chunk_metadata,
            load_parent_link_map_for_dir,
        )

        links = load_parent_link_map_for_dir(output_dir, doc_ids=doc_ids)
        enrich_leaf_chunk_metadata(chunks, links)

    if args.write_parent_pg and doc_ids:
        from retrieval.indexing.parent_store import index_parent_store

        counts = index_parent_store(
            input_dir=output_dir,
            doc_ids=doc_ids,
            rebuild=args.rebuild_stores,
        )
        print(
            "pg parent store indexed: "
            f"documents={counts['documents']} parent_chunks={counts['parent_chunks']} "
            f"leaf_chunks={counts['leaf_chunks']}"
        )

    if args.index_es_leaf and chunks:
        from collections import defaultdict

        from retrieval.indexing.es_index import index_nodes_to_elasticsearch

        by_category: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            cat = chunk.metadata.get("category") or chunk.metadata.get("doc_type")
            if cat:
                by_category[str(cat)].append(chunk)
        for cat, cat_chunks in by_category.items():
            indexed = index_nodes_to_elasticsearch(
                cat,
                chunks_to_nodes(cat_chunks),
                rebuild=args.rebuild_stores,
                require_configured=True,
            )
            print(f"es leaf indexed: category={cat} nodes={indexed}")

    if args.index_milvus_leaf and chunks:
        from retrieval.indexing.milvus_index import group_chunks_by_category, index_chunks_to_milvus

        counts = index_chunks_to_milvus(
            group_chunks_by_category(chunks),
            rebuild=args.rebuild_stores,
        )
        for cat, n in counts.items():
            print(f"milvus leaf indexed: category={cat} chunks={n}")

    if args.rebuild_index and chunks:
        from collections import defaultdict

        from retrieval.indexing.index import build_indexes_by_category
        from retrieval.core.kb_contract import SchemaGateError, validate_chunks_schema

        by_category: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            cat = chunk.metadata.get("category") or chunk.metadata.get("doc_type")
            if not cat:
                raise ValueError(f"chunk 缺少 category 元数据: {chunk.metadata}")
            if not args.skip_schema_gate:
                try:
                    validate_chunks_schema([chunk], str(cat))
                except SchemaGateError as exc:
                    raise SchemaGateError(
                        str(cat),
                        exc.missing_fields,
                        doc_id=str(chunk.metadata.get("doc_id") or ""),
                    ) from exc
            by_category[cat].append(chunk)

        nodes_by_category = {
            cat: chunks_to_nodes(cat_chunks) for cat, cat_chunks in by_category.items()
        }
        counts = build_indexes_by_category(
            nodes_by_category,
            rebuild=True,
            sync_elasticsearch=True,
        )
        for cat, n in counts.items():
            print(f"index rebuilt: category={cat} nodes={n} (pg + es if configured)")


if __name__ == "__main__":
    main()
