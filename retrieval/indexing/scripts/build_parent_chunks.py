"""从 cleaned chunks 生成 L2/L1 parent chunks。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TARGETS_BY_CATEGORY = {
    "annual_reports": {"l2": 1800, "l1": 5200},
    "research_reports": {"l2": 2200, "l1": 6000},
    "macro_research": {"l2": 2200, "l1": 6000},
    "industry_whitepapers": {"l2": 2000, "l1": 5600},
    "policy": {"l2": 2400, "l1": 6400},
}
DEFAULT_TARGETS = {"l2": 2000, "l1": 5600}


@dataclass
class LeafChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id") or "")

    @property
    def category(self) -> str:
        return str(self.metadata.get("category") or "")

    @property
    def section_path(self) -> str:
        return str(self.metadata.get("section_path") or self.metadata.get("section") or "未分类")

    @property
    def page_num(self) -> int | None:
        value = self.metadata.get("page_num")
        return value if isinstance(value, int) else None

    @property
    def chunk_index(self) -> int:
        value = self.metadata.get("chunk_index")
        return int(value) if value is not None else -1


@dataclass
class ParentBuffer:
    chunks: list[LeafChunk] = field(default_factory=list)

    @property
    def text_len(self) -> int:
        return sum(len(chunk.text) for chunk in self.chunks) + max(len(self.chunks) - 1, 0) * 2

    def append(self, chunk: LeafChunk) -> None:
        self.chunks.append(chunk)

    def clear(self) -> None:
        self.chunks.clear()


def _load_leaf_chunks(path: Path) -> list[LeafChunk]:
    chunks: list[LeafChunk] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            metadata = dict(obj.get("metadata") or {})
            doc_id = str(metadata.get("doc_id") or path.parent.name)
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                chunk_index = line_no - 1
                metadata["chunk_index"] = chunk_index
            chunk_id = f"{doc_id}:L3:{int(chunk_index):06d}"
            chunks.append(
                LeafChunk(
                    chunk_id=chunk_id,
                    text=str(obj.get("text") or ""),
                    metadata=metadata,
                )
            )
    return chunks


def _page_range(chunks: list[LeafChunk]) -> str:
    pages = [chunk.page_num for chunk in chunks if chunk.page_num is not None]
    if not pages:
        return ""
    start = min(pages)
    end = max(pages)
    return f"{start}" if start == end else f"{start}-{end}"


def _common_metadata(chunks: list[LeafChunk]) -> dict[str, Any]:
    first = chunks[0]
    metadata = first.metadata
    return {
        "format": metadata.get("format"),
        "doc_id": metadata.get("doc_id"),
        "title": metadata.get("title"),
        "category": metadata.get("category"),
        "source": metadata.get("source"),
        "file": metadata.get("file"),
        "authority_tier": metadata.get("authority_tier"),
        "doc_type": metadata.get("doc_type"),
        "doc_group": metadata.get("doc_group"),
        "issuer": metadata.get("issuer"),
        "effective_date": metadata.get("effective_date"),
        "ticker": metadata.get("ticker"),
        "fiscal_year": metadata.get("fiscal_year"),
        "language": metadata.get("language"),
    }


def _make_l2_record(
    chunks: list[LeafChunk],
    *,
    index: int,
) -> dict[str, Any]:
    first = chunks[0]
    parent_id = f"{first.doc_id}:L2:{index:06d}"
    sections = list(dict.fromkeys(chunk.section_path for chunk in chunks if chunk.section_path))
    section = " / ".join(sections)
    leaf_ids = [chunk.chunk_id for chunk in chunks]
    leaf_indices = [chunk.chunk_index for chunk in chunks]
    text = "\n\n".join(chunk.text for chunk in chunks).strip()
    metadata = {
        **{k: v for k, v in _common_metadata(chunks).items() if v not in (None, "")},
        "chunk_id": parent_id,
        "chunk_level": "L2",
        "parent_chunk_id": "",
        "root_chunk_id": "",
        "section_path": section,
        "section": section,
        "page_range": _page_range(chunks),
        "child_chunk_ids": leaf_ids,
        "child_chunk_indices": leaf_indices,
        "child_chunk_count": len(chunks),
        "block_types": sorted({str(chunk.metadata.get("block_type") or "") for chunk in chunks if chunk.metadata.get("block_type")}),
        "text_chars": len(text),
    }
    return {"text": text, "metadata": metadata}


def _make_l1_record(
    l2_records: list[dict[str, Any]],
    *,
    index: int,
) -> dict[str, Any]:
    first_meta = l2_records[0]["metadata"]
    doc_id = str(first_meta.get("doc_id") or "")
    root_id = f"{doc_id}:L1:{index:06d}"
    text = "\n\n".join(str(record.get("text") or "") for record in l2_records).strip()
    l2_ids = [str(record["metadata"].get("chunk_id")) for record in l2_records]
    leaf_ids: list[str] = []
    for record in l2_records:
        leaf_ids.extend(str(v) for v in record["metadata"].get("child_chunk_ids", []))

    metadata = {
        **{
            key: first_meta.get(key)
            for key in (
                "format",
                "doc_id",
                "title",
                "category",
                "source",
                "file",
                "authority_tier",
                "doc_type",
                "doc_group",
                "issuer",
                "effective_date",
                "ticker",
                "fiscal_year",
                "language",
            )
            if first_meta.get(key) not in (None, "")
        },
        "chunk_id": root_id,
        "chunk_level": "L1",
        "parent_chunk_id": "",
        "root_chunk_id": root_id,
        "section_path": " / ".join(dict.fromkeys(str(record["metadata"].get("section_path") or "") for record in l2_records if record["metadata"].get("section_path"))),
        "page_range": _merge_page_ranges(l2_records),
        "child_chunk_ids": l2_ids,
        "leaf_child_chunk_ids": leaf_ids,
        "child_chunk_count": len(l2_records),
        "leaf_child_chunk_count": len(leaf_ids),
        "text_chars": len(text),
    }
    return {"text": text, "metadata": metadata}


def _merge_page_ranges(records: list[dict[str, Any]]) -> str:
    pages: list[int] = []
    for record in records:
        page_range = str(record["metadata"].get("page_range") or "")
        for part in page_range.split("-"):
            if part.isdigit():
                pages.append(int(part))
    if not pages:
        return ""
    start = min(pages)
    end = max(pages)
    return f"{start}" if start == end else f"{start}-{end}"


def _build_l2(chunks: list[LeafChunk], target_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    buffer = ParentBuffer()
    current_section = ""

    def flush() -> None:
        if not buffer.chunks:
            return
        records.append(_make_l2_record(buffer.chunks, index=len(records)))
        buffer.clear()

    for chunk in chunks:
        section = chunk.section_path
        would_exceed = buffer.chunks and buffer.text_len + len(chunk.text) + 2 > target_chars
        section_changed = (
            buffer.chunks
            and section != current_section
            and buffer.text_len >= int(target_chars * 0.45)
        )
        if section_changed or would_exceed:
            flush()
        current_section = section
        buffer.append(chunk)
    flush()
    return records


def _build_l1(l2_records: list[dict[str, Any]], target_chars: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        records.append(_make_l1_record(buffer, index=len(records)))
        buffer = []
        buffer_len = 0

    for record in l2_records:
        text = str(record.get("text") or "")
        would_exceed = buffer and buffer_len + len(text) + 2 > target_chars
        if would_exceed:
            flush()
        buffer.append(record)
        buffer_len += len(text) + 2
    flush()
    return records


def build_parent_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    leaf_chunks = _load_leaf_chunks(chunks_path)
    if not leaf_chunks:
        return []
    category = leaf_chunks[0].category
    targets = TARGETS_BY_CATEGORY.get(category, DEFAULT_TARGETS)
    l2_records = _build_l2(leaf_chunks, int(targets["l2"]))
    l1_records = _build_l1(l2_records, int(targets["l1"]))

    l1_by_l2: dict[str, str] = {}
    for l1 in l1_records:
        root_id = str(l1["metadata"]["chunk_id"])
        for child_id in l1["metadata"].get("child_chunk_ids", []):
            l1_by_l2[str(child_id)] = root_id

    for l2 in l2_records:
        l2_id = str(l2["metadata"]["chunk_id"])
        root_id = l1_by_l2.get(l2_id, "")
        l2["metadata"]["parent_chunk_id"] = root_id
        l2["metadata"]["root_chunk_id"] = root_id

    return [*l1_records, *l2_records]


def write_parent_chunks(chunks_path: Path) -> tuple[Path, int]:
    records = build_parent_chunks(chunks_path)
    output_path = chunks_path.with_name("parent_chunks.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    return output_path, len(records)


def load_parent_link_map(parent_chunks_path: Path) -> dict[str, dict[str, str]]:
    """读取 parent_chunks.jsonl，生成 L3 chunk_id 到 L2/L1 的映射。"""
    links: dict[str, dict[str, str]] = {}
    with parent_chunks_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            if metadata.get("chunk_level") != "L2":
                continue
            l2_id = str(metadata.get("chunk_id") or "")
            root_id = str(metadata.get("root_chunk_id") or "")
            for child_id in metadata.get("child_chunk_ids", []):
                links[str(child_id)] = {
                    "parent_chunk_id": l2_id,
                    "root_chunk_id": root_id,
                }
    return links


def load_parent_link_map_for_dir(
    input_dir: Path,
    *,
    doc_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    links: dict[str, dict[str, str]] = {}
    for parent_path in sorted(input_dir.glob("**/parent_chunks.jsonl")):
        if doc_ids and parent_path.parent.name not in doc_ids:
            continue
        links.update(load_parent_link_map(parent_path))
    return links


def enrich_leaf_chunk_metadata(chunks: list[Any], links: dict[str, dict[str, str]]) -> None:
    """给内存中的 L3 chunk metadata 补稳定 chunk_id 与父块指针。"""
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        doc_id = str(metadata.get("doc_id") or "")
        chunk_index = int(metadata.get("chunk_index") or 0)
        chunk_id = str(metadata.get("chunk_id") or f"{doc_id}:L3:{chunk_index:06d}")
        metadata["chunk_id"] = chunk_id
        metadata["chunk_level"] = "L3"
        link = links.get(chunk_id) or {}
        if link.get("parent_chunk_id"):
            metadata["parent_chunk_id"] = link["parent_chunk_id"]
        if link.get("root_chunk_id"):
            metadata["root_chunk_id"] = link["root_chunk_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="从 cleaned_v2 chunks 生成 L1/L2 parent chunks")
    parser.add_argument("--input", type=Path, default=Path("knowledge/cleaned_v2"))
    parser.add_argument("--doc-id", nargs="+", default=None)
    args = parser.parse_args()

    allowed_doc_ids = set(args.doc_id or [])
    total_docs = 0
    total_parents = 0
    for chunks_path in sorted(args.input.glob("**/chunks.jsonl")):
        if allowed_doc_ids and chunks_path.parent.name not in allowed_doc_ids:
            continue
        output_path, count = write_parent_chunks(chunks_path)
        total_docs += 1
        total_parents += count
        print(f"{chunks_path.parent.name}: parents={count} output={output_path}")

    print(f"documents={total_docs} parent_chunks={total_parents}")


if __name__ == "__main__":
    main()
