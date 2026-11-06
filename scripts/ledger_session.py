"""把演示会话绑定到账房用户。模型不能在对话里改这个身份。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "app" / "backend"))

from retrieval.services.ledger_store import bind_actor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="绑定账房会话用户")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--user", required=True, help="U001 或 U002")
    parser.add_argument("--enterprise", default="", help="可选的默认企业，如 C001 或甲企业")
    args = parser.parse_args(argv)
    bind_actor(args.session_id, args.user, enterprise_code=args.enterprise or None)
    print(f"bound {args.session_id} -> {args.user.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
