---
name: index-data
description: 查询上证指数、沪深300、创业板指、恒生指数及其他指数行情和指标时使用。
required_tools:
  - iwencai.index.query
---

# 指数数据查询

- 仅通过 `iwencai.index.query` 查询指数事实。
- 区分指数名称、代码、交易日期和指标（点位、涨跌幅、成交量等）。
- 返回时保留数据日期和来源；没有数据就明确报告缺口。
