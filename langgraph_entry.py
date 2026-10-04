"""LangGraph Studio 入口。产品 HTTP 不加载本文件。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = str(ROOT / "app" / "backend")
ROOT_STR = str(ROOT)

# 仓库根下也有 app/（前端/后端目录），必须让 app/backend 排在最前，
# 否则 import app.core 会打到没有 core 的那个 app。
if BACKEND_DIR in sys.path:
    sys.path.remove(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

_app = sys.modules.get("app")
_app_file = str(getattr(_app, "__file__", "") or "").replace("\\", "/")
if _app is not None and "app/backend/app" not in _app_file:
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

from agents.finance_agent.financial_query_agent.workflows.predefined import (  # noqa: E402
    build_predefined_workflow_graph,
)
from agents.finance_agent.financial_query_agent.workflows.text_to_sql import (  # noqa: E402
    build_text_to_sql_workflow_graph,
)

from agents.finance_agent.pdf_agent import get_pdf_agent_graph  # noqa: E402

predefined_graph = build_predefined_workflow_graph().compile()
text_to_sql_graph = build_text_to_sql_workflow_graph().compile()
pdf_agent_graph = get_pdf_agent_graph()
