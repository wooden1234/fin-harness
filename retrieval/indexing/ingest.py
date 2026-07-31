from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = _SCRIPT_DIR.parent.parent
_BACKEND_DIR = ROOT_DIR / "app" / "backend"
if sys.path and os.path.abspath(sys.path[0]) == str(_SCRIPT_DIR):
    sys.path.pop(0)
for _path in (str(_BACKEND_DIR), str(ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

import re
from typing import List

import yaml

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from retrieval.clients.embeddings import get_embed_model
from retrieval.core.collections import get_table_name
from retrieval.indexing.index import EMBED_DIM, build_index

RAW_DIR = ROOT_DIR / "knowledge" / "raw"


def _guess_doc_type(filename: str) -> str:
    """根据文件名推断 doc_type（与 week-02 metadata 规范一致）。"""
    upper = filename.upper()
    if "FAQ" in upper:
        return "faq"
    if "POLICY" in upper:
        return "policy"
    return "regulation"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _extract_sections(full_text: str) -> list[tuple[int, str]]:
    """全文里所有 Markdown 标题及起始位置。"""
    return [(m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(full_text)]


def _section_for_offset(sections: list[tuple[int, str]], offset: int) -> str:
    """chunk 起始位置对应的最近标题。"""
    title = ""
    for pos, name in sections:
        if pos <= offset:
            title = name
        else:
            break
    return title or "未分类"


def enrich_section_metadata(nodes: list[TextNode], docs: list[Document]) -> None:
    """给每个 node 写入 section（最近的上级/当前标题）。"""
    doc_by_path = {d.metadata.get("file_path", ""): d for d in docs}

    for node in nodes:
        fp = node.metadata.get("file_path", "")
        doc = doc_by_path.get(fp)
        if not doc:
            node.metadata.setdefault("section", "未分类")
            continue

        full_text = doc.get_content(metadata_mode="none")
        sections = _extract_sections(full_text)
        offset = node.start_char_idx
        if offset is not None:
            node.metadata["section"] = _section_for_offset(sections, offset)
        else:
            node.metadata["section"] = "未分类"

def load_documents(file_path: Path) -> List[Document]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    docs: list[Document] = []
    for path in sorted(file_path.rglob("*.md")):
        if path.name.startswith("INDEX") or "_shared" in path.parts:
            continue
        raw = path.read_text(encoding="utf-8-sig")
        metadata: dict[str, object] = {}
        body = raw
        if raw.startswith("---\n"):
            end = raw.find("\n---", 4)
            if end < 0:
                raise ValueError(f"frontmatter_not_closed:{path}")
            parsed = yaml.safe_load(raw[4:end]) or {}
            if not isinstance(parsed, dict):
                raise ValueError(f"frontmatter_not_mapping:{path}")
            metadata.update(parsed)
            body = raw[end + 4 :].lstrip("\r\n")
        metadata.update(
            {
                "format": "md",
                "file_path": str(path),
                "source": path.name,
                "category": "faq",
            }
        )
        metadata.setdefault("doc_type", _guess_doc_type(path.name))
        docs.append(Document(text=body, metadata=metadata))
    return docs

# 在 _HEADING_RE 附近增加
_Q_HEADING_RE = re.compile(r"^### Q\d+.*$", re.MULTILINE)
_POLICY_SECTION_RE = re.compile(r"^(?:##\s+.+|### Q\d+.*)$", re.MULTILINE)


def _section_from_q_heading(line: str) -> str:
    """'### Q1：A股交易时间如何安排？' -> 'Q1：A股交易时间如何安排？'"""
    return re.sub(r"^###\s+", "", line.strip())


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[TextNode]:
    """FAQ 按完整问答切分，Policy 同时保留制度章节和问答块。"""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[TextNode] = []

    for doc in docs:
        text = doc.get_content(metadata_mode="none")
        doc_type = str(doc.metadata.get("doc_type") or "faq")
        doc_id = str(doc.metadata.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(
                f"faq_doc_id_required:{doc.metadata.get('file_path') or doc.metadata.get('source')}"
            )
        pattern = _POLICY_SECTION_RE if doc_type == "policy" else _Q_HEADING_RE
        matches = list(pattern.finditer(text))
        if not matches:
            continue

        base_meta = dict(doc.metadata)
        chunk_index = 0

        def append_node(node: TextNode) -> None:
            nonlocal chunk_index
            chunk_id = f"{doc_id}:faq:{chunk_index:06d}"
            node.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                }
            )
            node.id_ = chunk_id
            nodes.append(node)
            chunk_index += 1

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            q_text = text[start:end].strip()
            if not q_text:
                continue

            heading = m.group(0).strip()
            section = re.sub(r"^#{2,3}\s+", "", heading)
            chunk_kind = "qa_block" if heading.startswith("### Q") else "policy_section"
            meta = {**base_meta, "section": section, "chunk_kind": chunk_kind}
            relationships = (
                {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)}
                if doc_id != "None"
                else {}
            )

            if len(q_text) <= chunk_size:
                append_node(
                    TextNode(text=q_text, metadata=meta, relationships=relationships)
                )
            else:
                # 超长单题：只在题内切，避免跨题粘连
                sub_doc = Document(text=q_text, metadata=meta)
                for sub in splitter.get_nodes_from_documents([sub_doc]):
                    sub.metadata.update(meta)
                    if doc_id != "None":
                        sub.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                            node_id=doc_id
                        )
                    append_node(sub)

    return nodes


def validate_unique_chunk_ids(nodes: list[TextNode]) -> None:
    """在连接任何索引存储前拒绝缺失或重复的 chunk ID。"""
    seen: set[str] = set()
    duplicates: set[str] = set()
    missing: list[int] = []
    for position, node in enumerate(nodes):
        chunk_id = str(node.metadata.get("chunk_id") or "").strip()
        if not chunk_id:
            missing.append(position)
            continue
        if chunk_id in seen:
            duplicates.add(chunk_id)
        seen.add(chunk_id)
    if missing:
        raise ValueError(f"faq_chunk_id_missing:positions={missing[:10]}")
    if duplicates:
        raise ValueError(
            "faq_chunk_id_duplicate:ids=" + ",".join(sorted(duplicates)[:10])
        )


def index_faq_nodes(
    nodes: list[TextNode],
    *,
    rebuild: bool = False,
    index_es: bool = False,
    index_milvus: bool = False,
) -> dict[str, int]:
    """校验后将 FAQ 写入 PGVector，并按显式参数同步其他检索后端。"""
    validate_unique_chunk_ids(nodes)
    build_index(
        nodes,
        category="faq",
        rebuild=rebuild,
        sync_elasticsearch=False,
    )
    counts = {"pg": len(nodes), "es": 0, "milvus": 0}
    if index_es:
        from retrieval.indexing.es_index import index_nodes_to_elasticsearch

        counts["es"] = index_nodes_to_elasticsearch(
            "faq",
            nodes,
            rebuild=rebuild,
            require_configured=True,
        )
    if index_milvus:
        from retrieval.indexing.milvus_index import index_chunks_to_milvus

        indexed = index_chunks_to_milvus(
            {"faq": nodes},
            rebuild=rebuild,
        )
        counts["milvus"] = indexed.get("faq", 0)
    return counts


def run_ingest(raw_dir: Path | None = None) -> list[TextNode]:
    raw_dir = raw_dir or RAW_DIR
    docs = load_documents(raw_dir)
    nodes = chunk_documents(docs)
    if nodes:
        print("section:", nodes[0].metadata.get("section"))
    return nodes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--rebuild", action="store_true")  # Day 3 给 index 用
    parser.add_argument(
        "--index-es",
        action="store_true",
        help="将 FAQ chunks 同步写入 Elasticsearch",
    )
    parser.add_argument(
        "--index-milvus",
        action="store_true",
        help="将 FAQ chunks 同步写入 Milvus",
    )
    args = parser.parse_args()

    nodes = run_ingest(args.raw_dir)
    print(f"documents → nodes: {len(nodes)}")

    counts = index_faq_nodes(
        nodes,
        rebuild=args.rebuild,
        index_es=args.index_es,
        index_milvus=args.index_milvus,
    )
    print(
        f"index built → table={get_table_name('faq')}, dim={EMBED_DIM}, "
        f"pg={counts['pg']}, es={counts['es']}, milvus={counts['milvus']}"
    )

    for i, node in enumerate(nodes[:1]):  # 只看前 5 片
        print(f"\n--- chunk {i} ---")
        print("source:", node.metadata.get("source"))
        print("doc_type:", node.metadata.get("doc_type"))
        print("text_len:", len(node.text))
        print("text_preview:\n", node.text[:400], "...\n")
        # 入库后 node 上可能有 embedding；没有则再算一次（仅调试用）
        emb = getattr(node, "embedding", None)
        if emb is None:

            emb = get_embed_model().get_text_embedding(
                node.get_content(metadata_mode="none")  # 或 node.text
            )
        print("vector_dim:", len(emb))
        print("vector_head:", emb[:8])  

if __name__ == "__main__":
    main()
