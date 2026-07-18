"""LangGraph Studio / Agent Server 入口。

供 ``langgraph dev`` 加载各层 graph。不要在此传入 checkpointer——
LangGraph API 会自行管理 persistence（inmem / Postgres）。

在 Studio 左上角切换 graph 名称即可查看不同层级：
- fin_agent              主图
- finance_agent          编排子图（含 faq / pdf / financial_query / web_search）
- financial_query_agent  SQL 路由子图
- predefined_workflow    白名单 SQL 工作流
- text_to_sql_workflow   Text-to-SQL 工作流
- fin_agent_combined       全架构合图（主图 + finance + sql 一次看清）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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

graph = get_graph(with_checkpointer=False)
finance_graph = build_finance_agent_subgraph().compile()
financial_query_graph = build_financial_query_agent_graph().compile()
predefined_graph = build_predefined_workflow_graph().compile()
text_to_sql_graph = build_text_to_sql_workflow_graph().compile()
combined_graph = build_combined_overview_graph().compile()
