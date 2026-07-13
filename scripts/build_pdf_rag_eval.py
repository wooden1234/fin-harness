"""校验并导出 PDF RAG 评测集。

用法:
  python scripts/build_pdf_rag_eval.py          # 校验 + 写 jsonl
  python scripts/build_pdf_rag_eval.py --strict # 任一 keyword 缺失则失败
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from knowledge.eval.pdf_rag_eval_cases import PDF_RAG_EVAL_CASES  # noqa: E402

CLEANED = ROOT / "knowledge" / "cleaned"
OUT_PATH = ROOT / "knowledge" / "eval" / "pdf_rag_eval.jsonl"


def _load_doc_text(doc_id: str, category: str) -> str:
    path = CLEANED / category / doc_id / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts.append(json.loads(line)["text"])
    return "\n".join(parts)


def validate_cases(*, strict: bool = False) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    validated: list[dict] = []

    for i, case in enumerate(PDF_RAG_EVAL_CASES, start=1):
        qid = f"PDF-EVAL-{i:03d}"
        doc_id = case["expected_doc_id"]
        category = case["expected_category"]
        keywords = case["keywords"]

        try:
            blob = _load_doc_text(doc_id, category)
        except FileNotFoundError as exc:
            errors.append(f"{qid}: 找不到 chunks {exc}")
            continue

        missing = [kw for kw in keywords if kw not in blob]
        if missing:
            msg = f"{qid} [{doc_id}] 关键词未命中: {missing} | query={case['query'][:50]}"
            if strict:
                errors.append(msg)
            else:
                # 宽松模式：至少命中一个 keyword
                if not any(kw in blob for kw in keywords):
                    errors.append(msg)
                else:
                    errors.append(f"WARN {msg}")

        validated.append(
            {
                "query_id": qid,
                "query": case["query"],
                "expected_category": category,
                "expected_doc_id": doc_id,
                "keywords": keywords,
                "query_type": case.get("query_type", "factual"),
                "source_hint": case.get("source_hint", ""),
            }
        )

    if strict and errors:
        raise SystemExit("\n".join(errors))
    return validated, errors


def export_jsonl(cases: list[dict], path: Path = OUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="所有 keywords 必须命中")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    cases, issues = validate_cases(strict=args.strict)
    if len(cases) != 100:
        raise SystemExit(f"期望 100 条，实际 {len(cases)} 条")

    export_jsonl(cases, args.output)
    print(f"Wrote {len(cases)} cases → {args.output}")

    warns = [e for e in issues if e.startswith("WARN")]
    hard = [e for e in issues if not e.startswith("WARN")]
    if warns:
        print(f"\n{len(warns)} keyword partial-miss warnings:")
        for w in warns[:10]:
            print(" ", w)
        if len(warns) > 10:
            print(f"  ... and {len(warns) - 10} more")
    if hard:
        print(f"\n{len(hard)} errors:")
        for e in hard:
            print(" ", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
