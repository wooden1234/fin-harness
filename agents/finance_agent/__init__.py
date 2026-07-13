"""finance_agent：FinAgent 意图驱动编排子图。

职责：
1. planner 按用户意图拆分（非数据源类型）；
2. resolve_evidence 将意图映射为证据工具链；
3. 各 evidence worker（faq/pdf/financial_query/web_search）取证；
4. coverage gate：证据不足时沿链降级，禁止弱相关硬答；
5. join 收齐各分支结果；
6. summarize 汇总子任务结果。
"""

import importlib


def __getattr__(name):
    if name == "finance_agent":
        mod = importlib.import_module(
            "agents.finance_agent.graph"
        )
        return mod.build_finance_agent_subgraph().compile()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["finance_agent"]
