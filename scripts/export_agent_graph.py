"""导出 Agent 图结构（Mermaid / ASCII），便于 debug 与文档。

用法:
    python scripts/export_agent_graph.py
    python scripts/export_agent_graph.py --all
    python scripts/export_agent_graph.py --layer finance_agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.combined_overview_graph import build_combined_overview_graph  # noqa: E402
from agents.finance_agent.financial_query_agent.graph import (  # noqa: E402
    build_financial_query_agent_graph,
)
from agents.finance_agent.financial_query_agent.workflows.predefined import (  # noqa: E402
    build_predefined_workflow_graph,
)
from agents.finance_agent.financial_query_agent.workflows.text_to_sql import (  # noqa: E402
    build_text_to_sql_workflow_graph,
)
from agents.finance_agent.graph import build_finance_agent_subgraph  # noqa: E402
from agents.graph import get_graph  # noqa: E402

OUT_DIR = ROOT / "docs" / "architecture"

LAYERS: dict[str, tuple[Any, bool]] = {
    "fin_agent": (lambda: get_graph(with_checkpointer=False), False),
    "finance_agent": (lambda: build_finance_agent_subgraph().compile(), True),
    "financial_query_agent": (
        lambda: build_financial_query_agent_graph().compile(),
        False,
    ),
    "predefined_workflow": (
        lambda: build_predefined_workflow_graph().compile(),
        False,
    ),
    "text_to_sql_workflow": (
        lambda: build_text_to_sql_workflow_graph().compile(),
        False,
    ),
    "fin_agent_combined": (
        lambda: build_combined_overview_graph().compile(),
        False,
    ),
}


def _export_layer(name: str, *, xray: bool = False) -> Path:
    compiled = LAYERS[name][0]()
    drawable = compiled.get_graph(xray=xray)
    out = OUT_DIR / f"agent-graph-{name.replace('_', '-')}.mmd"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(drawable.draw_mermaid(), encoding="utf-8")
    return out


def _write_overview() -> Path:
    overview = OUT_DIR / "agent-graph-overview.md"
    overview.write_text(
        """# Agent 图结构总览

本项目 Agent 为 **5 层嵌套 LangGraph**，请按层级分别查看：

| 层级 | 文件 | Studio graph 名 | 说明 |
|------|------|-----------------|------|
| 1 | [agent-graph-fin-agent.mmd](./agent-graph-fin-agent.mmd) | `fin_agent` | 主图：guardrails → supervisor → plan_agent → final_answer |
| 2 | [agent-graph-finance-agent.mmd](./agent-graph-finance-agent.mmd) | `finance_agent` | 编排子图：plan → dispatch → faq/pdf/financial_query/web_search → join → summarize |
| 3 | [agent-graph-financial-query-agent.mmd](./agent-graph-financial-query-agent.mmd) | `financial_query_agent` | SQL 路由：planner → predefined / text_to_sql |
| 4 | [agent-graph-predefined-workflow.mmd](./agent-graph-predefined-workflow.mmd) | `predefined_workflow` | 白名单 SQL：init → select_tool → semantic → resolve → execute → format |
| 5 | [agent-graph-text-to-sql-workflow.mmd](./agent-graph-text-to-sql-workflow.mmd) | `text_to_sql_workflow` | Text-to-SQL：prepare → generate → validate → db_verify → format |
| **合图** | [agent-graph-fin-agent-combined.mmd](./agent-graph-fin-agent-combined.mmd) | `fin_agent_combined` | **全架构一张图**：主图 + finance + sql 所有关键节点 |

## LangSmith Studio

```bash
langgraph dev
```

打开 Studio 后，在左上角 **切换 graph 名称** 即可查看各层完整结构。

## 层级关系

```text
fin_agent
└── plan_agent (= finance_agent)
    ├── faq_agent          (单节点)
    ├── pdf_agent          (单节点)
    ├── financial_query_agent
    │   ├── planner
    │   ├── predefined  → 内部调用 predefined_workflow
    │   └── text_to_sql → 内部调用 text_to_sql_workflow
    └── web_search_agent   (单节点)
```

> `finance_agent` 导出使用了 `xray=1`，会把 `financial_query_agent` 内部节点展开显示。
""",
        encoding="utf-8",
    )
    return overview


def export_all() -> None:
    for name, (_, use_xray) in LAYERS.items():
        path = _export_layer(name, xray=use_xray)
        print(f"Wrote Mermaid → {path}")
    overview = _write_overview()
    print(f"Wrote overview → {overview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LangGraph agent graphs")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all layers to docs/architecture/",
    )
    parser.add_argument(
        "--layer",
        choices=sorted(LAYERS),
        default="fin_agent",
        help="Export a single layer (default: fin_agent)",
    )
    args = parser.parse_args()

    if args.all:
        export_all()
        return

    use_xray = LAYERS[args.layer][1]
    path = _export_layer(args.layer, xray=use_xray)
    print(f"Wrote Mermaid → {path}")

    compiled = LAYERS[args.layer][0]()
    try:
        print("\n--- ASCII preview ---")
        print(compiled.get_graph(xray=use_xray).draw_ascii())
    except ImportError as exc:
        print(f"\n(skip ASCII: {exc})")


if __name__ == "__main__":
    main()
