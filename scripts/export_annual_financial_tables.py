"""Export financial table chunks from cleaned annual report JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "knowledge" / "cleaned" / "annual_reports"
DEFAULT_JSONL = ROOT / "knowledge" / "cleaned" / "annual_financial_tables.jsonl"
DEFAULT_CSV = ROOT / "knowledge" / "cleaned" / "annual_financial_tables.csv"

PERIODIC_HEADER_KEYWORDS = (
    "本期数",
    "上年同期数",
    "本报告期",
    "上年同期",
    "本期发生额",
    "上期发生额",
    "本期發生額",
    "上期發生額",
    "期末余额",
    "期初余额",
    "期末餘額",
    "期初餘額",
    "年末",
    "期末",
    "期初",
    "同比",
    "增减",
    "增減",
    "变动比例",
    "變動比例",
    "比上年",
    "止年度",
    "年度",
    "十二月三十一日",
    "12月31日",
    "12 月 31 日",
    "季度",
    "月份",
    "三個月",
    "三个月",
    "第一季度",
    "第二季度",
    "第三季度",
    "第四季度",
)
PERIODIC_TABLE_KINDS = {
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "major_accounting_data",
}
SEGMENT_TABLE_KEYWORDS = (
    "主营业务分行业",
    "主营业务分产品",
    "主营业务分地区",
    "主营业务分销售模式",
    "主营业务分行业、分产品、分地区",
    "分行业情况",
    "分产品情况",
    "分地区情况",
    "分销售模式情况",
    "分行业",
    "分產品",
    "分产品",
    "分地区",
    "分部",
    "主营业务构成",
    "營業收入構成",
    "按分部劃分",
)
NOTE_TABLE_KEYWORDS = (
    "其他权益工具投资",
    "其他權益工具投資",
    "其他权益工具",
    "其他權益工具",
    "涉及政府补助",
    "政府补助",
    "政府補助",
    "资产及负债状况",
    "資產及負債狀況",
    "资产负债表项目说明",
    "資產負債表項目說明",
    "本公司財務狀況",
    "公允价值",
    "公允價值",
    "於聯營公司的投資",
    "于联营公司的投资",
    "联营公司的投资",
    "聯營公司的投資",
    "合营公司的投资",
    "合營公司的投資",
    "营业收入和营业成本情况",
    "营业收入和营业成本",
    "營業收入和營業成本",
    "境内外会计准则",
    "境內外會計準則",
    "募集资金",
    "募集資金",
    "应收款项融资",
    "應收款項融資",
    "已背书或贴现",
    "已背書或貼現",
    "背书或贴现",
    "背書或貼現",
    "营业外收入",
    "營業外收入",
    "关键技术或性能指标",
    "關鍵技術或性能指標",
)
COMPLEX_TABLE_KEYWORDS = (
    "所有者权益变动表",
    "所有者權益變動表",
    "股东权益变动表",
    "股東權益變動表",
    "权益变动表",
    "權益變動表",
    "合并所有者权益变动表",
    "合併所有者權益變動表",
    "母公司所有者权益变动表",
    "母公司所有者權益變動表",
)
EQUITY_CHANGE_COLUMN_KEYWORDS = (
    "实收资本",
    "實收資本",
    "资本公积",
    "資本公積",
    "减: 库存股",
    "减：库存股",
    "減：庫存股",
    "其他综合收益",
    "其他綜合收益",
    "专项储备",
    "專項儲備",
    "盈余公积",
    "盈餘公積",
    "未分配利润",
    "未分配利潤",
    "少数股东权益",
    "少數股東權益",
    "所有者权益合计",
    "所有者權益合計",
)


TABLE_KIND_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "balance_sheet",
        (
            "资产负债表",
            "資產負債表",
            "财务状况表",
            "財務狀況表",
            "资产总计",
            "資產總額",
            "负债合计",
            "負債總額",
            "所有者权益",
            "權益及負債",
        ),
    ),
    (
        "income_statement",
        (
            "利润表",
            "利潤表",
            "损益表",
            "損益表",
            "综合收益表",
            "綜合收益表",
            "全面收益表",
            "收入",
            "营业收入",
            "營業收入",
            "营业利润",
            "經營盈利",
            "净利润",
            "盈利",
            "每股收益",
        ),
    ),
    (
        "cash_flow_statement",
        (
            "现金流量表",
            "現金流量表",
            "现金流量净额",
            "現金流量淨額",
            "经营活动产生的现金流量",
            "經營活動產生的現金流量",
        ),
    ),
    (
        "major_accounting_data",
        ("主要会计数据", "主要會計數據", "主要财务指标", "主要財務指標", "近三年"),
    ),
    ("equity_changes", ("股东权益变动", "股東權益變動", "所有者权益变动", "權益變動")),
    (
        "segment_revenue",
        (
            "分行业",
            "分產品",
            "分地区",
            "分部",
            "主营业务",
            "營業收入構成",
            "按分部劃分",
        ),
    ),
    ("r_and_d", ("研发费用", "研發開支", "研发投入", "研發投入", "研发人员", "研發人員")),
    ("employee_compensation", ("应付职工薪酬", "應付職工薪酬", "职工薪酬", "員工")),
]


def _clean_table_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_markdown_row(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("|") or "|" not in line[1:]:
        return None
    if line.endswith("|"):
        line = line[:-1]
    cells = [cell.replace(r"\|", "|").strip() for cell in line[1:].split("|")]
    if len(cells) < 2:
        return None
    if all(not cell or re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
        return None
    return cells


def markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        row = split_markdown_row(line)
        if row:
            rows.append(row)
    return rows


def table_header_signature(text: str) -> str:
    rows = markdown_rows(text)
    if not rows:
        return ""
    return " | ".join(rows[0])


def row_has_periodic_header(row: list[str]) -> bool:
    cells = row[1:] if len(row) > 1 else row
    blob = " ".join(cells)
    if re.search(r"20\d{2}", blob):
        return True
    if re.search(r"[二〇零一二三四五六七八九]{4}年", blob):
        return True
    return any(keyword in blob for keyword in PERIODIC_HEADER_KEYWORDS)


def is_complex_table(text: str) -> bool:
    if any(keyword in text for keyword in COMPLEX_TABLE_KEYWORDS):
        return True
    if sum(1 for keyword in EQUITY_CHANGE_COLUMN_KEYWORDS if keyword in text) >= 4:
        return True
    return False


def is_segment_table(text: str) -> bool:
    return any(keyword in text for keyword in SEGMENT_TABLE_KEYWORDS)


def is_note_detail_table(section: str, text: str) -> bool:
    lines = (text or "").splitlines()
    header_context = "\n".join(lines[:5])
    haystack = f"{section}\n{header_context}"
    return any(keyword in haystack for keyword in NOTE_TABLE_KEYWORDS)


def classify_fact_parse_mode(table_kind: str, text: str, section: str = "") -> str:
    if is_complex_table(text):
        return "note_table"
    if is_segment_table(text):
        return "note_table"
    if is_note_detail_table(section, text):
        return "note_table"
    if table_kind not in PERIODIC_TABLE_KINDS:
        return "note_table"

    rows = markdown_rows(text)
    if not rows:
        return "unknown"
    header_scan = rows[:4]
    if any(row_has_periodic_header(row) for row in header_scan):
        return "periodic_fact"
    return "note_table"


def classify_table_kind(section: str, text: str) -> str:
    blob = f"{section} {text}"
    for kind, keywords in TABLE_KIND_KEYWORDS:
        if any(keyword in blob for keyword in keywords):
            return kind
    return "financial_other"


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _group_mode(rows: list[dict[str, Any]]) -> str:
    if any(is_complex_table(row.get("text") or "") for row in rows):
        return "note_table"
    if any(is_segment_table(row.get("text") or "") for row in rows):
        return "note_table"
    if any(is_note_detail_table(row.get("section") or "", row.get("text") or "") for row in rows):
        return "note_table"
    modes = {row.get("fact_parse_mode") for row in rows}
    if "periodic_fact" in modes:
        return "periodic_fact"
    if "note_table" in modes:
        return "note_table"
    return "unknown"


def unify_contiguous_table_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_key: tuple[Any, ...] | None = None
    prev_chunk_index: int | None = None

    for row in rows:
        chunk_index = _to_int(row.get("chunk_index"))
        key = (
            row.get("doc_id"),
            row.get("section"),
            row.get("table_header_signature"),
        )
        is_contiguous = (
            current
            and key == prev_key
            and chunk_index is not None
            and prev_chunk_index is not None
            and chunk_index == prev_chunk_index + 1
        )
        if not is_contiguous:
            if current:
                grouped.append(current)
            current = [row]
        else:
            current.append(row)
        prev_key = key
        prev_chunk_index = chunk_index

    if current:
        grouped.append(current)

    for group in grouped:
        if len(group) < 2:
            continue
        mode = _group_mode(group)
        for row in group:
            row["fact_parse_mode"] = mode
            row["fact_parse_group_size"] = len(group)
    return rows


def iter_financial_tables(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunks_path in sorted(input_dir.glob("*/chunks.jsonl")):
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                meta = obj.get("metadata") or {}
                if meta.get("block_type") != "table":
                    continue
                if meta.get("table_class") != "financial":
                    continue

                text = obj.get("text", "")
                section = meta.get("section_path") or meta.get("section") or ""
                table_kind = classify_table_kind(section, text)
                fact_parse_mode = classify_fact_parse_mode(table_kind, text, section)
                header_signature = table_header_signature(text)
                rows.append(
                    {
                        "doc_id": meta.get("doc_id", ""),
                        "title": meta.get("title", ""),
                        "ticker": meta.get("ticker", ""),
                        "fiscal_year": meta.get("fiscal_year", ""),
                        "source": meta.get("source", ""),
                        "page_num": meta.get("page_num", ""),
                        "chunk_index": meta.get("chunk_index", ""),
                        "section": section,
                        "table_kind": table_kind,
                        "fact_parse_mode": fact_parse_mode,
                        "fact_parse_group_size": 1,
                        "table_header_signature": header_signature,
                        "table_split_strategy": meta.get("table_split_strategy", ""),
                        "table_header_inherited": meta.get("table_header_inherited", ""),
                        "table_part_index": meta.get("table_part_index", ""),
                        "table_part_count": meta.get("table_part_count", ""),
                        "text": text,
                        "text_flat": _clean_table_text(text),
                    }
                )
    return unify_contiguous_table_modes(rows)


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id",
        "title",
        "ticker",
        "fiscal_year",
        "source",
        "page_num",
        "chunk_index",
        "section",
        "table_kind",
        "fact_parse_mode",
        "fact_parse_group_size",
        "table_header_signature",
        "table_split_strategy",
        "table_header_inherited",
        "table_part_index",
        "table_part_count",
        "text_flat",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_kind: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    for row in rows:
        by_kind[row["table_kind"]] = by_kind.get(row["table_kind"], 0) + 1
        by_mode[row["fact_parse_mode"]] = by_mode.get(row["fact_parse_mode"], 0) + 1
        by_doc[row["doc_id"]] = by_doc.get(row["doc_id"], 0) + 1

    print(f"financial_table_chunks={len(rows)}")
    print("by_fact_parse_mode:")
    for mode, count in sorted(by_mode.items()):
        print(f"  {mode}: {count}")
    print("by_kind:")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}")
    print("by_doc:")
    for doc_id, count in sorted(by_doc.items()):
        print(f"  {doc_id}: {count}")


def format_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export financial table chunks from cleaned annual reports."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows = iter_financial_tables(args.input_dir)
    write_jsonl(rows, args.jsonl_output)
    write_csv(rows, args.csv_output)
    print_summary(rows)
    print(f"jsonl={format_output_path(args.jsonl_output)}")
    print(f"csv={format_output_path(args.csv_output)}")


if __name__ == "__main__":
    main()
