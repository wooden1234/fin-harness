"""审计 cleaned chunks 的 metadata、长度分布和常见噪声。"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


REQUIRED_COMMON = {"format", "doc_id", "title", "category"}
REQUIRED_BY_CATEGORY = {
    "annual_reports": {"fiscal_year", "ticker"},
    "research_reports": {"issuer", "effective_date"},
    "industry_whitepapers": {"issuer", "effective_date"},
    "policy": {"effective_date"},
    "macro_research": {"effective_date"},
}

NOISE_PATTERNS = {
    "print_share": re.compile(r"\[打印\]|微博|微信|分享到|扫一扫"),
    "nav_bar": re.compile(r"首页|机构概况|新闻发布|政务信息|办事服务|互动交流|统计信息"),
    "toc_hint": re.compile(r"目录|更多信息请参阅第\d+页|P\d+"),
    "copyright": re.compile(r"免责声明|版权所有|Copyright|法律声明"),
}


def _percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return int(ordered[index])


def _missing_fields(metadata: dict[str, Any]) -> list[str]:
    category = str(metadata.get("category") or "")
    required = REQUIRED_COMMON | REQUIRED_BY_CATEGORY.get(category, set())
    return sorted(
        field for field in required if metadata.get(field) in (None, "", [], ())
    )


def audit_cleaned(input_dir: Path) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    seen_text: set[tuple[str, str]] = set()
    duplicate_count = 0
    sample_issues: list[dict[str, Any]] = []

    for chunks_path in sorted(input_dir.glob("**/chunks.jsonl")):
        with chunks_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                obj = json.loads(line)
                text = str(obj.get("text") or "")
                metadata = obj.get("metadata") or {}
                category = str(metadata.get("category") or chunks_path.parts[-3])

                category_stats = stats.setdefault(
                    category,
                    {
                        "files": set(),
                        "chunks": 0,
                        "empty_chunks": 0,
                        "missing_chunks": 0,
                        "short_lt80": 0,
                        "long_gt800": 0,
                        "long_gt1200": 0,
                        "block_types": {},
                        "lengths": [],
                        "noise": {name: 0 for name in NOISE_PATTERNS},
                    },
                )
                category_stats["files"].add(str(chunks_path.parent))
                category_stats["chunks"] += 1
                category_stats["lengths"].append(len(text))

                if not text.strip():
                    category_stats["empty_chunks"] += 1
                if len(text) < 80:
                    category_stats["short_lt80"] += 1
                if len(text) > 800:
                    category_stats["long_gt800"] += 1
                if len(text) > 1200:
                    category_stats["long_gt1200"] += 1

                block_type = str(metadata.get("block_type") or "unknown")
                category_stats["block_types"][block_type] = (
                    category_stats["block_types"].get(block_type, 0) + 1
                )

                missing = _missing_fields(metadata)
                if missing:
                    category_stats["missing_chunks"] += 1
                    if len(sample_issues) < 50:
                        sample_issues.append(
                            {
                                "type": "missing_metadata",
                                "path": str(chunks_path),
                                "line": line_no,
                                "doc_id": metadata.get("doc_id"),
                                "missing": missing,
                            }
                        )

                text_key = (str(metadata.get("doc_id") or ""), text.strip())
                if text_key in seen_text:
                    duplicate_count += 1
                else:
                    seen_text.add(text_key)

                for name, pattern in NOISE_PATTERNS.items():
                    if pattern.search(text):
                        category_stats["noise"][name] += 1

    normalized: dict[str, Any] = {}
    for category, category_stats in stats.items():
        lengths = category_stats.pop("lengths")
        files = category_stats.pop("files")
        normalized[category] = {
            **category_stats,
            "files": len(files),
            "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "p50_len": int(statistics.median(lengths)) if lengths else 0,
            "p90_len": _percentile(lengths, 90),
            "p95_len": _percentile(lengths, 95),
            "max_len": max(lengths) if lengths else 0,
        }

    return {
        "input_dir": str(input_dir),
        "categories": normalized,
        "duplicates_same_doc_text": duplicate_count,
        "sample_issues": sample_issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 cleaned chunks 质量")
    parser.add_argument("--input", type=Path, default=Path("knowledge/cleaned"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = audit_cleaned(args.input)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
