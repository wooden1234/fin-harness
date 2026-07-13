"""Build row-level annual financial facts from exported financial table chunks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "knowledge" / "cleaned" / "annual_financial_tables.jsonl"
DEFAULT_JSONL = ROOT / "knowledge" / "cleaned" / "annual_financial_facts.jsonl"
DEFAULT_CSV = ROOT / "knowledge" / "cleaned" / "annual_financial_facts.csv"

FIELDNAMES = [
    "doc_id",
    "title",
    "ticker",
    "fiscal_year",
    "source",
    "page_num",
    "chunk_index",
    "section",
    "table_kind",
    "row_index",
    "statement_type",
    "metric_name",
    "metric_alias",
    "period_label",
    "period_year",
    "period_type",
    "value",
    "raw_value",
    "unit",
    "currency",
    "raw_row",
    "raw_table_text",
]

_SEPARATOR_RE = re.compile(r"^-+$")
_NUMERIC_RE = re.compile(r"^[\(\-+]?\s*(?:[\d,]+(?:\.\d+)?|\.\d+)\s*\)?%?$")
_ARABIC_YEAR_RE = re.compile(r"(20\d{2})")
_ZH_YEAR_RE = re.compile(r"([二〇零一二三四五六七八九]{4})年")

_ZH_DIGITS = {
    "〇": "0",
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}

HEADER_FIRST_CELLS = {
    "项目",
    "項目",
    "科目",
    "主要会计数据",
    "主要會計數據",
    "主要财务指标",
    "主要財務指標",
    "非经常性损益项目",
    "非經常性損益項目",
}

HEADER_PERIOD_KEYWORDS = (
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
    "季度",
    "月份",
    "三個月",
    "三个月",
    "年度",
    "年末",
    "截至",
    "止年度",
    "十二月三十一日",
    "12月31日",
    "12 月 31 日",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse annual financial markdown tables into row-level facts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--include-raw-table",
        action="store_true",
        help="Keep full raw table text in every output row. CSV can become large.",
    )
    parser.add_argument(
        "--parse-all-tables",
        action="store_true",
        help="Parse all exported financial tables, including note_table rows. Default only parses periodic_fact.",
    )
    parser.add_argument(
        "--include-low-confidence",
        action="store_true",
        help="Keep empty/value_/unknown period labels for debugging. Default skips them.",
    )
    return parser.parse_args()


def split_markdown_row(line: str) -> list[str] | None:
    line = line.strip()
    if not line or "|" not in line:
        return None
    if line.startswith("【表格"):
        return None
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = [cell.replace(r"\|", "|").strip() for cell in line.split("|")]
    if len(cells) < 2:
        return None
    if all(not cell or _SEPARATOR_RE.match(cell.replace(" ", "")) for cell in cells):
        return None
    return cells


def extract_markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        row = split_markdown_row(line)
        if row:
            rows.append(row)
    return rows


def is_numeric_cell(value: str) -> bool:
    value = value.strip()
    if not value or value in {"-", "--", "—", "不适用", "不適用", "N/A"}:
        return False
    return bool(_NUMERIC_RE.match(value))


def parse_decimal(value: str) -> Decimal | None:
    raw = value.strip()
    if not is_numeric_cell(raw):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.strip("()").replace(",", "").replace("%", "").replace(" ", "")
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -parsed if negative else parsed


def looks_like_header(row: list[str]) -> bool:
    first_cell = row[0].strip() if row else ""
    value_cells = row[1:] if len(row) > 1 else row
    value_blob = " ".join(value_cells)
    full_blob = " ".join(row)

    if first_cell in HEADER_FIRST_CELLS:
        return True
    if any(token in value_blob for token in HEADER_PERIOD_KEYWORDS):
        return True
    if sum(1 for cell in value_cells if parse_period_year(cell) is not None) >= 1:
        return True
    if first_cell == "" and any(token in full_blob for token in HEADER_PERIOD_KEYWORDS):
        return True
    return False


def is_note_header(label: str) -> bool:
    return label.strip() in {"附注", "附註", "注释", "註釋", "附注(如适用)", "附註(如適用)"}


def is_period_header_cell(label: str) -> bool:
    label = label.strip()
    if not label:
        return False
    return classify_period(label) != "unknown" or parse_period_year(label) is not None


def count_specific_period_headers(headers: list[str], target_cols: list[int]) -> int:
    return sum(
        1
        for idx in target_cols
        if idx < len(headers) and parse_period_year(headers[idx]) is not None
    )


def update_headers(current: list[str] | None, row: list[str]) -> list[str]:
    if current is None:
        if parse_period_year(row[0]) is not None and sum(
            1 for cell in row if parse_period_year(cell) is not None
        ) >= 2:
            return [""] + row
        return row

    width = max(len(current), len(row))
    merged = value_headers(current, width)
    period_cells = [cell for cell in row if is_period_header_cell(cell)]
    if not period_cells:
        return row

    note_cols = {idx for idx, label in enumerate(merged) if is_note_header(label)}
    target_cols = [idx for idx in range(1, width) if idx not in note_cols]
    if len(period_cells) < len(target_cols) and count_specific_period_headers(merged, target_cols) >= len(target_cols):
        return merged
    if len(period_cells) <= len(target_cols):
        for idx, label in zip(target_cols, period_cells):
            merged[idx] = label
        return merged
    return row


def should_skip_metric(metric: str) -> bool:
    metric = metric.strip()
    if not metric:
        return True
    if metric in {"项目", "項目", "科目", "主要会计数据", "主要會計數據", "主要财务指标", "主要財務指標"}:
        return True
    if metric.endswith(":") or metric.endswith("："):
        return True
    if is_numeric_cell(metric):
        return True
    return False


def extract_unit_currency(text: str) -> tuple[str, str]:
    currency = ""
    unit = ""

    m = re.search(r"单位[:：]\s*([^\n|]+)", text)
    if m:
        unit_blob = m.group(1).strip()
        unit = unit_blob
        currency_match = re.search(r"[币幣]种[:：]\s*([^\s|]+)", unit_blob)
        if currency_match:
            currency = currency_match.group(1)

    if "人民幣百萬元" in text or "人民币百万元" in text:
        unit = "百万元"
        currency = "人民币"
    elif "人民幣千元" in text or "人民币千元" in text:
        unit = "千元"
        currency = "人民币"
    elif "人民幣萬元" in text or "人民币万元" in text:
        unit = "万元"
        currency = "人民币"
    elif "币种：人民币" in text or "币种： 人民币" in text or "幣種：人民幣" in text:
        currency = currency or "人民币"

    if "百万元" in unit or "百萬元" in unit:
        unit = "百万元"
    elif "万元" in unit or "萬元" in unit:
        unit = "万元"
    elif "千元" in unit:
        unit = "千元"
    elif "元" in unit:
        unit = "元"

    return unit, currency


def parse_period_year(label: str) -> int | None:
    m = _ARABIC_YEAR_RE.search(label)
    if m:
        return int(m.group(1))
    m = _ZH_YEAR_RE.search(label)
    if not m:
        return None
    digits = "".join(_ZH_DIGITS.get(ch, "") for ch in m.group(1))
    return int(digits) if len(digits) == 4 else None


def classify_period(label: str) -> str:
    if label.startswith("value_"):
        return "unknown"
    if any(token in label for token in ("同比", "增减", "變動", "变动", "%")):
        return "change_rate"
    if any(
        token in label
        for token in (
            "本期数",
            "上年同期数",
            "本报告期",
            "上年同期",
            "本期发生额",
            "上期发生额",
            "本期發生額",
            "上期發生額",
        )
    ):
        return "annual"
    if "止年度" in label or "年度" in label:
        return "annual"
    if any(
        token in label
        for token in (
            "年末",
            "期末余额",
            "期初余额",
            "期末餘額",
            "期初餘額",
            "12 月 31 日",
            "12月31日",
            "十二月三十一日",
            "於",
        )
    ):
        return "period_end"
    if any(token in label for token in ("季度", "月份", "三個月", "三个月")):
        return "quarter"
    if parse_period_year(label) is not None:
        return "annual"
    return "unknown"


def infer_period_year(label: str, fiscal_year: Any) -> int | None:
    year = parse_period_year(label)
    if year is not None:
        return year
    try:
        fy = int(fiscal_year)
    except (TypeError, ValueError):
        return None
    if any(token in label for token in ("本期", "本年", "本年度", "期末余额", "期末餘額")):
        return fy
    if any(token in label for token in ("上年", "去年", "期初余额", "期初餘額")):
        return fy - 1
    return None


def infer_statement_type(table: dict[str, Any]) -> str:
    section = table.get("section") or ""
    kind = table.get("table_kind") or ""
    if section:
        return section
    return kind


def value_headers(headers: list[str], width: int) -> list[str]:
    if len(headers) >= width:
        return headers[:width]
    return headers + [f"value_{i}" for i in range(len(headers), width)]


def row_to_facts(
    table: dict[str, Any],
    row: list[str],
    headers: list[str],
    row_index: int,
    *,
    include_low_confidence: bool = False,
) -> list[dict[str, Any]]:
    width = max(len(row), len(headers))
    cells = row + [""] * (width - len(row))
    hdrs = value_headers(headers, width)
    metric = cells[0].strip()
    if should_skip_metric(metric):
        return []

    unit, currency = extract_unit_currency(table.get("text") or "")
    facts: list[dict[str, Any]] = []
    for col_idx, raw_value in enumerate(cells[1:], start=1):
        value = parse_decimal(raw_value)
        if value is None:
            continue
        period_label = hdrs[col_idx].strip() if col_idx < len(hdrs) else f"value_{col_idx}"
        if is_note_header(period_label):
            continue
        period_type = classify_period(period_label)
        if (
            not include_low_confidence
            and (not period_label or period_label.startswith("value_") or period_type == "unknown")
        ):
            continue
        fact_unit = "%" if period_type == "change_rate" or raw_value.strip().endswith("%") else unit
        period_year = infer_period_year(period_label, table.get("fiscal_year"))
        facts.append(
            {
                "doc_id": table.get("doc_id", ""),
                "title": table.get("title", ""),
                "ticker": table.get("ticker", ""),
                "fiscal_year": table.get("fiscal_year", ""),
                "source": table.get("source", ""),
                "page_num": table.get("page_num", ""),
                "chunk_index": table.get("chunk_index", ""),
                "section": table.get("section", ""),
                "table_kind": table.get("table_kind", ""),
                "row_index": row_index,
                "statement_type": infer_statement_type(table),
                "metric_name": metric,
                "metric_alias": "",
                "period_label": period_label,
                "period_year": period_year or "",
                "period_type": period_type,
                "value": str(value),
                "raw_value": raw_value.strip(),
                "unit": fact_unit,
                "currency": currency,
                "raw_row": " | ".join(cells).strip(),
                "raw_table_text": table.get("text", ""),
            }
        )
    return facts


def parse_table(
    table: dict[str, Any],
    *,
    include_low_confidence: bool = False,
    initial_headers: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str] | None]:
    rows = extract_markdown_rows(table.get("text") or "")
    if not rows:
        return [], initial_headers

    facts: list[dict[str, Any]] = []
    headers: list[str] | None = list(initial_headers) if initial_headers else None
    for row_index, row in enumerate(rows):
        if looks_like_header(row):
            headers = update_headers(headers, row)
            continue
        if headers is None:
            headers = ["metric"] + [f"value_{i}" for i in range(1, len(row))]
        facts.extend(
            row_to_facts(
                table,
                row,
                headers,
                row_index,
                include_low_confidence=include_low_confidence,
            )
        )
    return facts, headers


def load_tables(path: Path) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tables.append(json.loads(line))
    return tables


def write_jsonl(rows: list[dict[str, Any]], output: Path, *, include_raw_table: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = row if include_raw_table else {**row, "raw_table_text": ""}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], output: Path, *, include_raw_table: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            obj = row if include_raw_table else {**row, "raw_table_text": ""}
            writer.writerow({name: obj.get(name, "") for name in FIELDNAMES})


def print_summary(tables: list[dict[str, Any]], parsed_tables: list[dict[str, Any]], facts: list[dict[str, Any]]) -> None:
    by_doc: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    parsed_by_mode: dict[str, int] = {}
    skipped_by_mode: dict[str, int] = {}
    for table in tables:
        mode = table.get("fact_parse_mode") or "legacy_unspecified"
        by_mode[mode] = by_mode.get(mode, 0) + 1
    parsed_ids = {id(table) for table in parsed_tables}
    for table in tables:
        mode = table.get("fact_parse_mode") or "legacy_unspecified"
        target = parsed_by_mode if id(table) in parsed_ids else skipped_by_mode
        target[mode] = target.get(mode, 0) + 1
    for fact in facts:
        by_doc[fact["doc_id"]] = by_doc.get(fact["doc_id"], 0) + 1
        by_kind[fact["table_kind"]] = by_kind.get(fact["table_kind"], 0) + 1

    print(f"tables={len(tables)}")
    print(f"parsed_tables={len(parsed_tables)}")
    print(f"skipped_tables={len(tables) - len(parsed_tables)}")
    print("tables_by_fact_parse_mode:")
    for key, count in sorted(by_mode.items()):
        print(f"  {key}: {count}")
    print("parsed_tables_by_mode:")
    for key, count in sorted(parsed_by_mode.items()):
        print(f"  {key}: {count}")
    print("skipped_tables_by_mode:")
    for key, count in sorted(skipped_by_mode.items()):
        print(f"  {key}: {count}")
    print(f"facts={len(facts)}")
    print("facts_by_kind:")
    for key, count in sorted(by_kind.items()):
        print(f"  {key}: {count}")
    print("facts_by_doc:")
    for key, count in sorted(by_doc.items()):
        print(f"  {key}: {count}")


def format_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    tables = load_tables(args.input)
    facts: list[dict[str, Any]] = []
    parsed_tables: list[dict[str, Any]] = []
    header_cache: dict[tuple[Any, Any, Any], list[str]] = {}
    for table in tables:
        if not args.parse_all_tables and table.get("fact_parse_mode") and table.get("fact_parse_mode") != "periodic_fact":
            continue
        parsed_tables.append(table)
        header_key = (
            table.get("doc_id"),
            table.get("section"),
            table.get("table_header_signature"),
        )
        table_facts, final_headers = parse_table(
            table,
            include_low_confidence=args.include_low_confidence,
            initial_headers=header_cache.get(header_key),
        )
        facts.extend(table_facts)
        if final_headers:
            header_cache[header_key] = final_headers

    write_jsonl(facts, args.jsonl_output, include_raw_table=args.include_raw_table)
    write_csv(facts, args.csv_output, include_raw_table=args.include_raw_table)
    print_summary(tables, parsed_tables, facts)
    print(f"jsonl={format_output_path(args.jsonl_output)}")
    print(f"csv={format_output_path(args.csv_output)}")


if __name__ == "__main__":
    main()
