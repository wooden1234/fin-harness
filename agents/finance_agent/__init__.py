"""finance_agent：多 worker 金融检索子图。

职责与 assistgen 的多工具工作流保持一致：
1. planner 先做跨能力任务拆分；
2. 各 worker 分别执行 faq / pdf / financial_query / web_search；
3. join 收齐各分支结果；
4. summarize 汇总子任务结果。

⚠️ 惰性加载：避免 state_mixins.py 导入本包下的 state.py 时
   触发 graph.py → states.py 循环。
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
