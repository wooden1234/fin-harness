"""LangGraph Studio 仅保留 SQL 调试图；产品 HTTP 不加载本文件。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "app" / "backend"
for path in (str(BACKEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.finance_agent.financial_query_agent.workflows.predefined import (  # noqa: E402
    build_predefined_workflow_graph,
)
from agents.finance_agent.financial_query_agent.workflows.text_to_sql import (  # noqa: E402
    build_text_to_sql_workflow_graph,
)

predefined_graph = build_predefined_workflow_graph().compile()
text_to_sql_graph = build_text_to_sql_workflow_graph().compile()
