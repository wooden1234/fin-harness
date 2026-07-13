"""导出主图结构（Mermaid / ASCII），便于 debug。

用法:
    source .venv/bin/activate
    python scripts/export_agent_graph.py
    python scripts/export_agent_graph.py --out docs/architecture/agent-graph.mmd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents.graph import get_graph  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "architecture" / "agent-graph.mmd"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LangGraph agent graph")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Mermaid output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    graph = get_graph(with_checkpointer=False)
    drawable = graph.get_graph()
    mermaid = drawable.draw_mermaid()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(mermaid, encoding="utf-8")
    print(f"Wrote Mermaid → {args.out}")
    try:
        print("\n--- ASCII preview ---")
        print(drawable.draw_ascii())
    except ImportError as exc:
        print(f"\n(skip ASCII: {exc})")


if __name__ == "__main__":
    main()
