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
import sys


def __getattr__(name):
    if name == "finance_agent":
        mod = importlib.import_module(
            "agents.finance_agent.graph"
        )
        return mod.build_finance_agent_subgraph().compile()
    if name == "financial_query_agent":
        # 历史 app.agents shim 可能覆盖父包属性，始终返回 canonical 子包。
        canonical_name = "agents.finance_agent.financial_query_agent"
        return sys.modules.get(canonical_name) or importlib.import_module(
            canonical_name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["finance_agent"]
