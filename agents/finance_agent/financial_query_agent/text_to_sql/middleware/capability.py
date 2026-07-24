"""text_to_sql 数据库能力硬边界。"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from agents.finance_agent.financial_query_agent.services.schemas import (
    GeneratedFinancialSql,
)
from agents.finance_agent.financial_query_agent.text_to_sql.middleware.base import (
    MiddlewareResult,
)
from agents.finance_agent.financial_query_agent.text_to_sql.state import (
    TextToSqlState,
)

_REALTIME_MARKER_RE = re.compile(
    r"当前|实时|今日|今天|现价|最新价|盘中|本周|近\s*\d+\s*(?:日|天)"
)
_MARKET_DATA_RE = re.compile(
    r"股价|行情|涨跌幅?|资金流|成交量|成交额|换手率|盘口|K线|"
    r"MACD|RSI|市盈率|市净率"
)
_DOCUMENT_MARKER_RE = re.compile(
    r"年报|季报|半年报|公告|招股书|研报|白皮书|政策文件|附注|原文|披露"
)
_DOCUMENT_DETAIL_RE = re.compile(
    r"原因|依据|说明|风险|策略|审计意见|公司治理|坏账准备|计提比例|"
    r"分产品|分地区|市场份额|销量|用户数|人员占比|股东持股|担保|"
    r"诉讼|关联交易|客户集中度"
)


def database_capability_reason(question: str) -> str:
    """只识别明确超出财务事实库的硬边界，模糊问题继续交给 SQL 生成器。"""
    normalized = question.strip()
    if _REALTIME_MARKER_RE.search(normalized) and _MARKET_DATA_RE.search(normalized):
        return "realtime_market_data_not_supported"
    if _DOCUMENT_MARKER_RE.search(normalized) and _DOCUMENT_DETAIL_RE.search(normalized):
        return "document_detail_not_supported"
    return ""


class DatabaseCapabilityMiddleware:
    """在调用 SQL 生成模型前拦截明确不属于结构化事实库的问题。"""

    async def before_generate(
        self,
        state: TextToSqlState,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        del config
        reason = database_capability_reason(state["question"])
        if not reason:
            return None
        return MiddlewareResult(
            halt=True,
            halt_reason="unsupported",
            state_updates={"route_reason": reason},
        )

    async def after_generate(
        self,
        state: TextToSqlState,
        generated: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        del state, generated, config
        return None

    async def after_correct(
        self,
        state: TextToSqlState,
        corrected: GeneratedFinancialSql,
        config: RunnableConfig | None = None,
    ) -> MiddlewareResult | None:
        del state, corrected, config
        return None


__all__ = ["DatabaseCapabilityMiddleware", "database_capability_reason"]
