---
name: market-quotes
description: 查询股票、ETF、指数的实时或历史行情、涨跌幅、成交量、资金流向和技术指标时使用。
required_tools:
  - iwencai.market.query
---

# 行情数据查询

- 仅通过 `iwencai.market.query` 获取行情事实，不使用模型记忆补齐数值。
- 将用户的标的、时间范围、指标和复权要求原样转换为问财自然语言条件。
- 返回数据时注明查询条件、时间口径和数据来源；空结果不得编造。
- 需要排序或条件过滤时，将标准化 `CandidateSet` 交给确定性执行器 `market.compute`。
