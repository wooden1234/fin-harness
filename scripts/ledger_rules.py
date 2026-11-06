"""管理账房规则文件：列出、检索、停用、启用和导入。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app" / "backend"))

from retrieval.services.ledger_rule_kb import search_rule_kb, sync_rule_knowledge
from retrieval.services.ledger_rules import (
    import_rule_file,
    load_rules,
    set_rule_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="管理账房规则 Markdown")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="列出全部规则版本")
    list_cmd.add_argument("--active-only", action="store_true")

    search_cmd = sub.add_parser("search", help="按企业、月份和状态筛选")
    search_cmd.add_argument("--enterprise", required=True)
    search_cmd.add_argument("--month", required=True)
    search_cmd.add_argument("--topic", default="")
    search_cmd.add_argument("--include-disabled", action="store_true")

    disable_cmd = sub.add_parser("disable", help="停用一个规则版本")
    disable_cmd.add_argument("--rule-id", required=True)
    disable_cmd.add_argument("--version", required=True)

    enable_cmd = sub.add_parser("enable", help="启用一个规则版本")
    enable_cmd.add_argument("--rule-id", required=True)
    enable_cmd.add_argument("--version", required=True)

    import_cmd = sub.add_parser("import", help="导入一篇规则 Markdown")
    import_cmd.add_argument("path", type=Path)

    sub.add_parser("sync", help="把 Markdown 规则同步进 ledger.rules 知识库")

    args = parser.parse_args(argv)
    if args.command == "list":
        rules = [rule for rule in load_rules() if rule.status == "active" or not args.active_only]
        _print_rules(rules)
        return 0
    if args.command == "search":
        rules = search_rule_kb(
            enterprise=args.enterprise,
            month=args.month,
            topic=args.topic or None,
            include_disabled=args.include_disabled,
        )
        print(json.dumps([rule.public_dict() for rule in rules], ensure_ascii=False, indent=2))
        return 0
    if args.command == "sync":
        print(f"synced {sync_rule_knowledge()} rules")
        return 0
    if args.command == "disable":
        path = set_rule_status(args.rule_id, args.version, "disabled")
        sync_rule_knowledge()
        print(path)
        return 0
    if args.command == "enable":
        path = set_rule_status(args.rule_id, args.version, "active")
        sync_rule_knowledge()
        print(path)
        return 0
    dest = import_rule_file(args.path)
    sync_rule_knowledge()
    print(dest)
    return 0


def _print_rules(rules) -> None:
    for rule in rules:
        enterprises = ",".join(rule.enterprises)
        print(
            f"{rule.rule_id}\t{rule.version}\t{rule.status}\t{enterprises}\t"
            f"{rule.effective_from}\t{rule.name}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
