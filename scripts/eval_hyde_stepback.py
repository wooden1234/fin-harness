"""HyDE / Step-back 检索消融评测。

不要走完整 PDF Agent 图：证据路由会把「改写收益」和「是否触发改写」缠在一起。
本脚本固定五路检索，只跟线上 retrieve 对齐：hybrid + admit + 原问/改写 RRF 融合。

用法：
  # 1) 从 ES 抽候选 chunk，人工改成评测集
  python scripts/eval_hyde_stepback.py export --out retrieval/eval/rewrite_candidates.jsonl

  # 2) 按桶跑原问 / 改写单路 / 融合
  python scripts/eval_hyde_stepback.py run --cases retrieval/eval/hyde_stepback.jsonl

  # 3) 只测原问基线（不调改写 LLM）
  python scripts/eval_hyde_stepback.py run --cases retrieval/eval/hyde_stepback.jsonl --arms original --skip-cosine

线上 admit_pdf_hits 当前会把多数 PDF 滤成 0 条，改写评测默认不加 --admit。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
load_dotenv(ROOT / ".env")

from app.core.config import settings  # noqa: E402
from agents.finance_agent.pdf_agent.query_rewrite.hyde import hyde_node  # noqa: E402
from agents.finance_agent.pdf_agent.query_rewrite.step_back import step_back_node  # noqa: E402
from agents.finance_agent.pdf_agent.retrieval.multi_query_fuse import (  # noqa: E402
    fuse_original_and_rewrite_hits,
)
from retrieval import RetrievalHit, get_pdf_retriever  # noqa: E402
from retrieval.clients.embeddings import aget_query_embedding_cached, get_embed_model  # noqa: E402
from retrieval.clients.es_client import create_es_client, index_name  # noqa: E402
from retrieval.core.collections import pdf_categories  # noqa: E402
from retrieval.services import admit_pdf_hits  # noqa: E402

ARMS = ("original", "step_back_only", "hyde_only", "fused_sb", "fused_hyde")
K_VALUES = (5, 10)
PDF_INDEXES = (
    "annual_reports",
    "macro_research",
    "research_reports",
    "industry_whitepapers",
    "policy",
)
_EXPORT_QUERIES = (
    "销售毛利率",
    "净利润 原因 OR 下滑 OR 亏损",
    "影响 机制 OR 趋势",
    "营业收入",
    "风险因素",
)


def _norm_chunk_id(value: Any) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text.lstrip("0") or "0"


def gold_keys(case: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for item in case.get("relevant_chunks") or []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        chunk_id = _norm_chunk_id(item.get("chunk_id") or item.get("chunk_index") or item.get("node_id"))
        if doc_id and chunk_id:
            keys.add((doc_id, chunk_id))
    return keys


def hit_key(hit: RetrievalHit) -> tuple[str, str]:
    doc_id = str(hit.metadata.get("doc_id") or "").strip()
    chunk_id = hit.metadata.get("chunk_index")
    if chunk_id in (None, ""):
        chunk_id = hit.node_id or hit.metadata.get("chunk_id") or ""
    return doc_id, _norm_chunk_id(chunk_id)


def hit_keys(hits: list[RetrievalHit]) -> list[tuple[str, str]]:
    return [hit_key(item) for item in hits]


def hit_at_k(keys: list[tuple[str, str]], gold: set[tuple[str, str]], k: int) -> float:
    return 1.0 if gold.intersection(keys[:k]) else 0.0


def recall_at_k(keys: list[tuple[str, str]], gold: set[tuple[str, str]], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(keys[:k])) / len(gold)


def mrr(keys: list[tuple[str, str]], gold: set[tuple[str, str]]) -> float:
    for rank, key in enumerate(keys, 1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(keys: list[tuple[str, str]], gold: set[tuple[str, str]], k: int) -> float:
    dcg = 0.0
    for rank, key in enumerate(keys[:k], 1):
        if key in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def entity_keep(query: str, rewrite: str, entities: list[str]) -> float:
    required = [item for item in entities if str(item).strip()]
    if not required:
        return 1.0
    text = str(rewrite or "")
    return sum(1.0 for item in required if item in text) / len(required)


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    n1 = math.sqrt(sum(a * a for a in left))
    n2 = math.sqrt(sum(b * b for b in right))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not str(row.get("query") or "").strip():
            raise ValueError(f"{path}:{line_no} 需要非空 query")
        rows.append(row)
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def cmd_export(args: argparse.Namespace) -> int:
    es = create_es_client()
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    per_query = max(int(args.per_query), 1)
    for category in PDF_INDEXES:
        name = index_name(category)
        for query in _EXPORT_QUERIES:
            response = es.search(
                index=name,
                size=per_query,
                source=[
                    "doc_id",
                    "node_id",
                    "chunk_index",
                    "issuer",
                    "ticker",
                    "fiscal_year",
                    "year",
                    "section",
                    "leaf_text",
                    "text",
                    "parent_chunk_id",
                ],
                query={"query_string": {"query": query, "fields": ["leaf_text", "text", "section"]}},
            )
            for hit in response.get("hits", {}).get("hits", []):
                src = hit.get("_source") or {}
                doc_id = str(src.get("doc_id") or "").strip()
                node_id = str(src.get("node_id") or "").strip()
                chunk_id = _norm_chunk_id(src.get("chunk_index") if src.get("chunk_index") is not None else node_id)
                key = (doc_id, chunk_id)
                if not doc_id or key in seen:
                    continue
                seen.add(key)
                text = str(src.get("leaf_text") or src.get("text") or "").replace("\n", " ").strip()
                rows.append(
                    {
                        "id": f"{category}-{len(rows)+1:03d}",
                        "bucket": "unlabeled",
                        "query": "",
                        "expect_strategy": "none",
                        "must_keep_entities": [
                            item
                            for item in (
                                src.get("issuer"),
                                src.get("ticker"),
                                str(src.get("fiscal_year") or src.get("year") or ""),
                            )
                            if item
                        ],
                        "categories": [category],
                        "relevant_chunks": [
                            {
                                "doc_id": doc_id,
                                "chunk_id": chunk_id,
                                "node_id": node_id,
                            }
                        ],
                        "preview": text[:400],
                        "section": src.get("section"),
                    }
                )
    dump_jsonl(Path(args.out), rows)
    print(f"exported {len(rows)} candidates -> {args.out}")
    print("标注：填 query / bucket(too_narrow|vocab_gap|control) / expect_strategy / must_keep_entities")
    return 0


def _select_arms(case: dict[str, Any], requested: set[str], mode: str) -> list[str]:
    if "original" not in requested and mode != "full":
        requested = {"original", *requested}
    if mode == "full":
        return [arm for arm in ARMS if arm in requested]
    bucket = str(case.get("bucket") or case.get("expect_strategy") or "").strip()
    expect = str(case.get("expect_strategy") or "").strip()
    selected = ["original"]
    if bucket in {"too_narrow", "step_back"} or expect == "step_back":
        selected.extend(["step_back_only", "fused_sb"])
    if bucket in {"vocab_gap", "hyde"} or expect == "hyde":
        selected.extend(["hyde_only", "fused_hyde"])
    if bucket in {"control", "none"} or expect == "none":
        selected.extend(["step_back_only", "hyde_only", "fused_sb", "fused_hyde"])
    return [arm for arm in ARMS if arm in selected and arm in requested]


async def _rewrite(strategy: str, query: str) -> str:
    state = {"original_query": query, "query": query, "rewrite_count": 0}
    node = step_back_node if strategy == "step_back" else hyde_node
    result = await node(state)
    rewritten = str(result.get("rewrite_query") or result.get("query") or "").strip()
    return rewritten or query


async def _search(
    retriever: Any,
    query: str,
    *,
    top_k: int,
    admit: bool,
) -> list[RetrievalHit]:
    hits = await retriever.asearch(query, top_k=top_k)
    if admit:
        return admit_pdf_hits(hits, use_mode="direct_qa")
    return hits


async def _fuse(
    retriever: Any,
    *,
    original_query: str,
    original_hits: list[RetrievalHit],
    rewrite_hits: list[RetrievalHit],
    top_k: int,
    admit: bool,
) -> list[RetrievalHit]:
    fused = await fuse_original_and_rewrite_hits(
        original_query=original_query,
        original_hits=original_hits,
        rewrite_hits=rewrite_hits,
        retriever=retriever,
        top_k=top_k,
    )
    if admit:
        return admit_pdf_hits(fused.hits, use_mode="direct_qa")
    return fused.hits


async def _gold_text(case: dict[str, Any]) -> str:
    chunks = [item for item in (case.get("relevant_chunks") or []) if isinstance(item, dict)]
    if not chunks:
        return ""
    first = chunks[0]
    node_id = str(first.get("node_id") or "").strip()
    doc_id = str(first.get("doc_id") or "").strip()
    categories = list(case.get("categories") or pdf_categories())
    es = create_es_client()
    for category in categories:
        body: dict[str, Any]
        if node_id:
            body = {"term": {"node_id": node_id}}
        else:
            body = {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {"term": {"chunk_index": int(_norm_chunk_id(first.get("chunk_id")))}},
                    ]
                }
            }
        response = es.search(index=index_name(category), size=1, query=body)
        hits = response.get("hits", {}).get("hits") or []
        if not hits:
            continue
        src = hits[0].get("_source") or {}
        return str(src.get("leaf_text") or src.get("text") or "").strip()
    return ""


def _score_arm(
    *,
    hits: list[RetrievalHit],
    gold: set[tuple[str, str]],
    original_keys: list[tuple[str, str]] | None = None,
) -> dict[str, float | int]:
    keys = hit_keys(hits)
    original = original_keys or []
    rewrite_gold = set(keys) & gold
    orig_gold = set(original) & gold
    row: dict[str, float | int] = {
        "hit_count": len(hits),
        "mrr": mrr(keys, gold),
        "novel_gold": len(rewrite_gold - orig_gold) if original else 0,
        "lost_gold": len(orig_gold - rewrite_gold) if original else 0,
        "pollution5": (
            len(set(keys[:5]) - gold - set(original[:5])) / 5.0 if original else 0.0
        ),
    }
    for k in K_VALUES:
        row[f"hit@{k}"] = hit_at_k(keys, gold, k)
        row[f"recall@{k}"] = recall_at_k(keys, gold, k)
        row[f"ndcg@{k}"] = ndcg_at_k(keys, gold, k)
    return row


def _retriever_for_case(case: dict[str, Any], *, top_k: int) -> Any:
    categories = [str(item) for item in (case.get("categories") or []) if str(item)]
    return get_pdf_retriever(
        categories=categories or None,
        top_k=top_k,
        similarity_threshold=None,
        hybrid=True,
        rerank_min_score=settings.RERANK_MIN_SCORE,
    )


async def _eval_case(
    case: dict[str, Any],
    *,
    top_k: int,
    arms: set[str],
    mode: str,
    with_cosine: bool,
    admit: bool,
) -> dict[str, Any]:
    query = str(case.get("query") or "").strip()
    gold = gold_keys(case)
    selected = _select_arms(case, arms, mode)
    retriever = _retriever_for_case(case, top_k=top_k)
    started = time.perf_counter()

    original_hits: list[RetrievalHit] = []
    if any(arm in selected for arm in ARMS):
        original_hits = await _search(retriever, query, top_k=top_k, admit=admit)
    original_keys = hit_keys(original_hits)

    rewrites: dict[str, str] = {}
    rewrite_hits: dict[str, list[RetrievalHit]] = {}
    need_sb = any(arm in selected for arm in ("step_back_only", "fused_sb"))
    need_hyde = any(arm in selected for arm in ("hyde_only", "fused_hyde"))
    if need_sb:
        rewrites["step_back"] = await _rewrite("step_back", query)
    if need_hyde:
        rewrites["hyde"] = await _rewrite("hyde", query)

    rerank_enabled = bool(getattr(retriever, "rerank_enabled", False))
    if need_sb or need_hyde:
        retriever.rerank_enabled = False
        try:
            if need_sb:
                rewrite_hits["step_back"] = await _search(
                    retriever, rewrites["step_back"], top_k=top_k, admit=admit
                )
            if need_hyde:
                rewrite_hits["hyde"] = await _search(
                    retriever, rewrites["hyde"], top_k=top_k, admit=admit
                )
        finally:
            retriever.rerank_enabled = rerank_enabled

    fused: dict[str, list[RetrievalHit]] = {}
    if "fused_sb" in selected:
        fused["fused_sb"] = await _fuse(
            retriever,
            original_query=query,
            original_hits=original_hits,
            rewrite_hits=rewrite_hits.get("step_back") or [],
            top_k=top_k,
            admit=admit,
        )
    if "fused_hyde" in selected:
        fused["fused_hyde"] = await _fuse(
            retriever,
            original_query=query,
            original_hits=original_hits,
            rewrite_hits=rewrite_hits.get("hyde") or [],
            top_k=top_k,
            admit=admit,
        )

    arm_hits = {
        "original": original_hits,
        "step_back_only": rewrite_hits.get("step_back") or [],
        "hyde_only": rewrite_hits.get("hyde") or [],
        "fused_sb": fused.get("fused_sb") or [],
        "fused_hyde": fused.get("fused_hyde") or [],
    }
    entities = [str(item) for item in (case.get("must_keep_entities") or [])]
    arm_metrics: dict[str, Any] = {}
    for arm in selected:
        metrics = _score_arm(
            hits=arm_hits[arm],
            gold=gold,
            original_keys=original_keys if arm != "original" else None,
        )
        if arm == "step_back_only":
            metrics["entity_keep"] = entity_keep(query, rewrites.get("step_back", ""), entities)
        elif arm == "hyde_only":
            metrics["entity_keep"] = entity_keep(query, rewrites.get("hyde", ""), entities)
        elif arm == "fused_sb":
            metrics["entity_keep"] = entity_keep(query, rewrites.get("step_back", ""), entities)
        elif arm == "fused_hyde":
            metrics["entity_keep"] = entity_keep(query, rewrites.get("hyde", ""), entities)
        else:
            metrics["entity_keep"] = 1.0
        arm_metrics[arm] = metrics

    delta_cos = None
    if with_cosine and rewrites.get("hyde") and gold:
        try:
            gold_text = await _gold_text(case)
            if gold_text:
                embed = get_embed_model()
                q_vec, h_vec, g_vec = await asyncio.gather(
                    aget_query_embedding_cached(query),
                    aget_query_embedding_cached(rewrites["hyde"]),
                    embed.aget_text_embedding(gold_text[:4000]),
                )
                delta_cos = cosine(h_vec, g_vec) - cosine(q_vec, g_vec)
        except Exception as exc:  # noqa: BLE001
            delta_cos = None
            arm_metrics.setdefault("hyde_only", {})["cosine_error"] = type(exc).__name__

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "id": case.get("id") or query[:32],
        "bucket": case.get("bucket") or "unlabeled",
        "query": query,
        "expect_strategy": case.get("expect_strategy") or "none",
        "rewrites": rewrites,
        "delta_cos": delta_cos,
        "latency_ms": elapsed_ms,
        "gold_count": len(gold),
        "arms": arm_metrics,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("bucket") or "unlabeled")
        for arm, metrics in (row.get("arms") or {}).items():
            grouped[(bucket, arm)].append(
                {
                    **metrics,
                    "delta_cos": row.get("delta_cos") if "hyde" in arm else None,
                }
            )
    summary = []
    for (bucket, arm), items in sorted(grouped.items()):
        rec: dict[str, Any] = {"bucket": bucket, "arm": arm, "n": len(items)}
        for field in (
            "hit@5",
            "hit@10",
            "ndcg@5",
            "ndcg@10",
            "recall@5",
            "recall@10",
            "mrr",
            "entity_keep",
            "novel_gold",
            "lost_gold",
            "pollution5",
        ):
            rec[field] = round(_mean([float(item.get(field) or 0.0) for item in items]), 4)
        cos_values = [float(item["delta_cos"]) for item in items if item.get("delta_cos") is not None]
        rec["delta_cos"] = round(_mean(cos_values), 4) if cos_values else None
        summary.append(rec)
    return summary


def _add_deltas(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary:
        by_bucket[row["bucket"]][row["arm"]] = row
    extra = []
    for bucket, arms in by_bucket.items():
        baseline = arms.get("original")
        if not baseline:
            continue
        for arm in ("fused_sb", "fused_hyde", "step_back_only", "hyde_only"):
            current = arms.get(arm)
            if not current:
                continue
            extra.append(
                {
                    "bucket": bucket,
                    "arm": f"Δ {arm} - original",
                    "n": min(current["n"], baseline["n"]),
                    **{
                        field: round(current[field] - baseline[field], 4)
                        for field in ("hit@5", "hit@10", "ndcg@5", "recall@10")
                    },
                    "entity_keep": current.get("entity_keep"),
                    "novel_gold": current.get("novel_gold"),
                    "lost_gold": current.get("lost_gold"),
                    "delta_cos": current.get("delta_cos"),
                }
            )
    return summary + extra


def _print_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "bucket",
        "arm",
        "n",
        "hit@5",
        "hit@10",
        "ndcg@5",
        "recall@10",
        "entity_keep",
        "novel_gold",
        "lost_gold",
        "delta_cos",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join("" if row.get(field) is None else str(row.get(field)) for field in fields))


async def cmd_run(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    cases = load_jsonl(cases_path)
    if args.limit:
        cases = cases[: max(int(args.limit), 1)]
    requested = set(args.arms) if args.arms else set(ARMS)
    top_k = max(int(args.top_k or settings.PDF_RETRIEVAL_TOP_K or 10), 5)
    results = []
    for case in cases:
        row = await _eval_case(
            case,
            top_k=top_k,
            arms=requested,
            mode=args.mode,
            with_cosine=not args.skip_cosine,
            admit=args.admit,
        )
        results.append(row)
        print(
            f"{row['id']}\t{row['bucket']}\t"
            + "\t".join(
                f"{arm} hit@5={metrics.get('hit@5')}"
                for arm, metrics in row["arms"].items()
            )
        )
    summary = _add_deltas(_summarize(results))
    print("\n=== summary ===")
    _print_table(summary)
    payload = {"top_k": top_k, "cases": results, "summary": summary}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyDE / Step-back 检索量化评测")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="从 ES 导出待标注 chunk")
    export_p.add_argument("--out", default=str(ROOT / "retrieval" / "eval" / "rewrite_candidates.jsonl"))
    export_p.add_argument("--per-query", type=int, default=2)

    run_p = sub.add_parser("run", help="按黄金集跑消融")
    run_p.add_argument("--cases", default=str(ROOT / "retrieval" / "eval" / "hyde_stepback.jsonl"))
    run_p.add_argument("--out", default=str(ROOT / "retrieval" / "eval" / "hyde_stepback_report.json"))
    run_p.add_argument("--arms", nargs="+", choices=ARMS)
    run_p.add_argument("--mode", choices=("expected", "full"), default="expected")
    run_p.add_argument("--top-k", type=int, default=0)
    run_p.add_argument("--limit", type=int, default=0)
    run_p.add_argument("--skip-cosine", action="store_true")
    run_p.add_argument(
        "--admit",
        action="store_true",
        help="套用线上 admit_pdf_hits；默认关闭，避免 manifest 把改写收益测成全 0",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "export":
        return cmd_export(args)
    return asyncio.run(cmd_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
