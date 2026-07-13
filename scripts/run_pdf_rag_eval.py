"""运行 knowledge/eval/pdf_rag_eval.jsonl，评测 PDF RAG 检索效果。"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

EVAL_PATH = ROOT / "knowledge" / "eval" / "pdf_rag_eval.jsonl"
DEFAULT_TOP_K = 3


def load_eval_cases(path: Path = EVAL_PATH) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _doc_match(hit, case: dict) -> bool:
    return (hit.metadata or {}).get("doc_id") == case.get("expected_doc_id")


def _category_match(hit, case: dict) -> bool:
    cat = case.get("expected_category", "")
    return hit.category == cat or (hit.metadata or {}).get("category") == cat


def _keyword_match(text: str, keywords: list[str]) -> bool:
    return bool(keywords) and any(kw in (text or "") for kw in keywords)


def _all_keywords_match(text: str, keywords: list[str]) -> bool:
    return bool(keywords) and all(kw in (text or "") for kw in keywords)


def _first_rank(flags: list[bool]) -> int | None:
    for i, ok in enumerate(flags, start=1):
        if ok:
            return i
    return None


def _reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _summarize(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "top_k": top_k,
        "doc_at_1": sum(bool(r["doc_at_1"]) for r in rows),
        "doc_at_k": sum(bool(r["doc_at_k"]) for r in rows),
        "category_at_k": sum(bool(r["category_at_k"]) for r in rows),
        "keyword_any_at_k": sum(bool(r["keyword_any_at_k"]) for r in rows),
        "keyword_all_at_k": sum(bool(r["keyword_all_at_k"]) for r in rows),
        "relevant_at_k": sum(bool(r["relevant_at_k"]) for r in rows),
        "mrr_doc": sum(float(r["doc_rr"]) for r in rows) / total if total else 0.0,
        "mrr_keyword_any": sum(float(r["keyword_any_rr"]) for r in rows) / total if total else 0.0,
    }


def _group_summaries(rows: list[dict[str, Any]], *, top_k: int, key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: _summarize(items, top_k=top_k) for name, items in sorted(grouped.items())}


def _print_summary(title: str, summary: dict[str, Any]) -> None:
    total = int(summary["total"])
    top_k = int(summary["top_k"])
    print(f"\n{title} (n={total}, k={top_k})")
    print(f"  Doc@1:        {_rate(summary['doc_at_1'], total):.1%} ({summary['doc_at_1']}/{total})")
    print(f"  Doc@{top_k}:        {_rate(summary['doc_at_k'], total):.1%} ({summary['doc_at_k']}/{total})")
    print(f"  Category@{top_k}:   {_rate(summary['category_at_k'], total):.1%} ({summary['category_at_k']}/{total})")
    print(
        f"  KeywordAny@{top_k}: {_rate(summary['keyword_any_at_k'], total):.1%} "
        f"({summary['keyword_any_at_k']}/{total})"
    )
    print(
        f"  KeywordAll@{top_k}: {_rate(summary['keyword_all_at_k'], total):.1%} "
        f"({summary['keyword_all_at_k']}/{total})"
    )
    print(f"  Relevant@{top_k}:   {_rate(summary['relevant_at_k'], total):.1%} ({summary['relevant_at_k']}/{total})")
    print(f"  MRR Doc:       {summary['mrr_doc']:.4f}")
    print(f"  MRR Keyword:   {summary['mrr_keyword_any']:.4f}")


def run_eval(
    *,
    eval_path: Path = EVAL_PATH,
    top_k: int = DEFAULT_TOP_K,
    pdf_only: bool = True,
    categories: list[str] | None = None,
) -> dict:
    cases = load_eval_cases(eval_path)
    from app.retrieval import get_pdf_retriever, get_retriever

    if pdf_only:
        retriever = get_pdf_retriever(categories=categories, top_k=top_k, similarity_threshold=None)
    else:
        retriever = get_retriever(categories=categories, top_k=top_k, similarity_threshold=None)

    rows: list[dict] = []

    for case in cases:
        query = case["query"]
        hits = retriever.search(query, top_k=top_k)
        padded = hits + [None] * (top_k - len(hits))

        hit_fields: dict = {}
        doc_flags: list[bool] = []
        category_flags: list[bool] = []
        keyword_any_flags: list[bool] = []
        keyword_all_flags: list[bool] = []
        relevant_flags: list[bool] = []

        for j, h in enumerate(padded[:top_k], start=1):
            if h is None:
                doc_flags.append(False)
                category_flags.append(False)
                keyword_any_flags.append(False)
                keyword_all_flags.append(False)
                relevant_flags.append(False)
                hit_fields.update(
                    {
                        f"hit{j}_score": "",
                        f"hit{j}_doc_id": "",
                        f"hit{j}_category": "",
                        f"hit{j}_preview": "",
                        f"hit{j}_doc_match": False,
                        f"hit{j}_keyword_match": False,
                        f"hit{j}_category_match": False,
                        f"hit{j}_keyword_any_match": False,
                        f"hit{j}_keyword_all_match": False,
                        f"hit{j}_relevant": False,
                    }
                )
                continue

            doc_ok = _doc_match(h, case)
            category_ok = _category_match(h, case)
            kw_any_ok = _keyword_match(h.text, case.get("keywords") or [])
            kw_all_ok = _all_keywords_match(h.text, case.get("keywords") or [])
            relevant_ok = doc_ok or (category_ok and kw_any_ok)
            doc_flags.append(doc_ok)
            category_flags.append(category_ok)
            keyword_any_flags.append(kw_any_ok)
            keyword_all_flags.append(kw_all_ok)
            relevant_flags.append(relevant_ok)

            preview = (h.text or "").replace("\n", " ")[:120]
            hit_fields[f"hit{j}_score"] = round(h.score, 4)
            hit_fields[f"hit{j}_doc_id"] = (h.metadata or {}).get("doc_id", "")
            hit_fields[f"hit{j}_category"] = h.category or (h.metadata or {}).get("category", "")
            hit_fields[f"hit{j}_preview"] = preview
            hit_fields[f"hit{j}_doc_match"] = doc_ok
            hit_fields[f"hit{j}_keyword_match"] = kw_any_ok
            hit_fields[f"hit{j}_category_match"] = category_ok
            hit_fields[f"hit{j}_keyword_any_match"] = kw_any_ok
            hit_fields[f"hit{j}_keyword_all_match"] = kw_all_ok
            hit_fields[f"hit{j}_relevant"] = relevant_ok

        doc_rank = _first_rank(doc_flags)
        category_rank = _first_rank(category_flags)
        keyword_any_rank = _first_rank(keyword_any_flags)
        keyword_all_rank = _first_rank(keyword_all_flags)
        relevant_rank = _first_rank(relevant_flags)

        rows.append(
            {
                "query_id": case.get("query_id"),
                "query": query,
                "query_type": case.get("query_type", ""),
                "expected_category": case.get("expected_category", ""),
                "expected_doc_id": case.get("expected_doc_id", ""),
                "source_hint": case.get("source_hint", ""),
                "doc_rank": doc_rank or "",
                "category_rank": category_rank or "",
                "keyword_any_rank": keyword_any_rank or "",
                "keyword_all_rank": keyword_all_rank or "",
                "relevant_rank": relevant_rank or "",
                "doc_rr": round(_reciprocal_rank(doc_rank), 4),
                "keyword_any_rr": round(_reciprocal_rank(keyword_any_rank), 4),
                "doc_at_1": bool(doc_flags and doc_flags[0]),
                "doc_at_k": bool(doc_rank),
                "category_at_k": bool(category_rank),
                "keyword_any_at_k": bool(keyword_any_rank),
                "keyword_all_at_k": bool(keyword_all_rank),
                "relevant_at_k": bool(relevant_rank),
                # Backward-compatible columns used by older CSV readers.
                "top3_doc_match": bool(doc_rank) if top_k == 3 else "",
                "top3_relevant": bool(relevant_rank) if top_k == 3 else "",
                "top1_doc_match": bool(doc_flags and doc_flags[0]),
                **hit_fields,
            }
        )

        print(
            f"[{case.get('query_id')}] doc@{top_k}={bool(doc_rank)} "
            f"kw@{top_k}={bool(keyword_any_rank)} rel@{top_k}={bool(relevant_rank)} "
            f"doc@1={rows[-1]['doc_at_1']} "
            f"| {query[:36]}..."
        )

    summary = _summarize(rows, top_k=top_k)
    return {
        "summary": summary,
        "by_query_type": _group_summaries(rows, top_k=top_k, key="query_type"),
        "by_category": _group_summaries(rows, top_k=top_k, key="expected_category"),
        "rows": rows,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PDF RAG 检索评测")
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--include-faq", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "knowledge" / "eval" / "pdf_rag_eval_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "knowledge" / "eval" / "pdf_rag_eval_summary.json",
    )
    args = parser.parse_args()

    result = run_eval(
        eval_path=args.eval,
        top_k=args.top_k,
        pdf_only=not args.include_faq,
        categories=args.categories,
    )
    summary = result["summary"]
    rows = result["rows"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(
            {
                "summary": result["summary"],
                "by_query_type": result["by_query_type"],
                "by_category": result["by_category"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.summary_output}")
    _print_summary("Overall", summary)

    print("\nBy query_type")
    for name, group_summary in result["by_query_type"].items():
        _print_summary(name, group_summary)

    print("\nBy category")
    for name, group_summary in result["by_category"].items():
        _print_summary(name, group_summary)


if __name__ == "__main__":
    main()
