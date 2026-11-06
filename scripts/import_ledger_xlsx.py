"""从比赛示例 Excel 导入账房业务数据。重复导入保留已保存的用途和备注。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app" / "backend"))

from retrieval.services.ledger_store import import_excel

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入账房发票和银行流水")
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--reset", action="store_true", help="按 Excel 重建，覆盖已保存的用途和备注")
    args = parser.parse_args(argv)
    xlsx = args.xlsx or _default_xlsx()
    if args.db is not None:
        counts = import_excel(xlsx, db_path=args.db, reset=args.reset)
        print(f"imported into {args.db}")
    else:
        counts = import_excel(xlsx, reset=args.reset)
        print("imported into PostgreSQL schema ledger")
    for key, value in counts.items():
        print(f"{key}: {value}")
    return 0


def _default_xlsx() -> Path:
    matches = sorted(_ROOT.glob("赛题二*/02_示例数据.xlsx"))
    if not matches:
        raise SystemExit("未找到示例数据 Excel")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
