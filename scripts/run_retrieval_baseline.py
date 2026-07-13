"""生成 knowledge/eval/retrieval_baseline.csv（20 条 query + Top3 检索结果）。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.retrieval import get_faq_retriever  # noqa: E402

# query, 期望来源文件, 期望命中关键词（任一出现在 Top3 文本即视为 relevant）
BASELINE_QUERIES: list[tuple[str, str, list[str]]] = [
    ("A股交易时间如何安排？", "01_Stock_Trading_Rules_FAQ.md", ["交易时间", "9:30", "集合竞价"]),
    ("什么是 T+1 交易制度？", "01_Stock_Trading_Rules_FAQ.md", ["T+1", "下一个交易日"]),
    ("涨跌停板制度是怎样规定的？", "01_Stock_Trading_Rules_FAQ.md", ["涨跌幅", "10%", "20%"]),
    ("集合竞价的成交原则是什么？", "01_Stock_Trading_Rules_FAQ.md", ["集合竞价", "最大成交量"]),
    ("股票交易的最小报价单位是多少？", "01_Stock_Trading_Rules_FAQ.md", ["报价单位", "0.01"]),
    ("A股交易需要缴纳哪些费用？", "01_Stock_Trading_Rules_FAQ.md", ["佣金", "印花税", "过户费"]),
    ("港股通交易有哪些特殊规则？", "01_Stock_Trading_Rules_FAQ.md", ["港股通", "T+0"]),
    ("什么是证券投资基金？", "02_Fund_Investment_Guide_FAQ.md", ["证券投资基金", "基金份额"]),
    ("公募基金和私募基金有何区别？", "02_Fund_Investment_Guide_FAQ.md", ["公募", "私募"]),
    ("基金申购赎回的基本流程是怎样的？", "02_Fund_Investment_Guide_FAQ.md", ["申购", "赎回", "T+"]),
    ("基金定投是什么？有什么优势？", "02_Fund_Investment_Guide_FAQ.md", ["定投", "平均成本"]),
    ("ETF 与普通开放式基金有什么不同？", "02_Fund_Investment_Guide_FAQ.md", ["ETF", "交易所"]),
    ("期货交易与股票交易有何本质区别？", "03_Futures_Trading_Policy_FAQ.md", ["保证金", "杠杆", "期货"]),
    ("期货保证金制度如何运作？", "03_Futures_Trading_Policy_FAQ.md", ["保证金", "追加"]),
    ("期货交易时间是怎样的？", "03_Futures_Trading_Policy_FAQ.md", ["交易时间", "夜盘"]),
    ("什么是穿仓？如何防范？", "03_Futures_Trading_Policy_FAQ.md", ["穿仓", "风险控制"]),
    ("什么是债券？债券的基本要素有哪些？", "04_Bond_Market_Basics_FAQ.md", ["债券", "面值", "票面利率"]),
    ("债券价格与收益率是什么关系？", "04_Bond_Market_Basics_FAQ.md", ["价格", "收益率", "反向"]),
    ("可转债有什么特点？", "04_Bond_Market_Basics_FAQ.md", ["可转债", "转股"]),
    ("债券投资面临哪些风险？", "04_Bond_Market_Basics_FAQ.md", ["信用风险", "利率风险", "流动性"]),
]


def _is_relevant(hit_text: str, hit_source: str, expected_source: str, keywords: list[str]) -> bool:
    if expected_source and expected_source in (hit_source or ""):
        return True
    text = hit_text or ""
    return any(kw in text for kw in keywords)


def main() -> None:
    out_dir = ROOT / "knowledge" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_baseline.csv"

    retriever = get_faq_retriever(top_k=3, similarity_threshold=None)

    fieldnames = [
        "query_id",
        "query",
        "expected_source",
        "top3_relevant",
        "hit1_score",
        "hit1_source",
        "hit1_section",
        "hit1_preview",
        "hit2_score",
        "hit2_source",
        "hit2_section",
        "hit2_preview",
        "hit3_score",
        "hit3_source",
        "hit3_section",
        "hit3_preview",
    ]

    rows: list[dict[str, str | int | float | bool]] = []
    relevant_count = 0

    for i, (query, expected_source, keywords) in enumerate(BASELINE_QUERIES, start=1):
        hits = retriever.search(query, top_k=3)
        padded = hits + [None] * (3 - len(hits))

        hit_fields: dict[str, str | float] = {}
        any_relevant = False
        for j, h in enumerate(padded[:3], start=1):
            if h is None:
                hit_fields.update(
                    {
                        f"hit{j}_score": "",
                        f"hit{j}_source": "",
                        f"hit{j}_section": "",
                        f"hit{j}_preview": "",
                    }
                )
                continue
            preview = (h.text or "").replace("\n", " ")[:120]
            src = h.metadata.get("source", "") if h.metadata else ""
            section = h.metadata.get("section", "") if h.metadata else ""
            hit_fields[f"hit{j}_score"] = round(h.score, 4)
            hit_fields[f"hit{j}_source"] = src
            hit_fields[f"hit{j}_section"] = section
            hit_fields[f"hit{j}_preview"] = preview
            if _is_relevant(h.text, src, expected_source, keywords):
                any_relevant = True

        if any_relevant:
            relevant_count += 1

        rows.append(
            {
                "query_id": i,
                "query": query,
                "expected_source": expected_source,
                "top3_relevant": any_relevant,
                **hit_fields,
            }
        )
        print(f"[{i:02d}] relevant={any_relevant} hits={len(hits)} | {query[:30]}...")

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rate = relevant_count / len(BASELINE_QUERIES) if BASELINE_QUERIES else 0
    print(f"\nWrote {out_path}")
    print(f"Top3 relevant: {relevant_count}/{len(BASELINE_QUERIES)} ({rate:.0%})")


if __name__ == "__main__":
    main()
