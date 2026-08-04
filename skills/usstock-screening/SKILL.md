---
name: usstock-screening
description: 通过自然语言查询进行美股筛选，支持行情指标、财务指标、行业概念、业绩预测、研报评级等多条件组合筛选。返回符合条件的相关美股数据。当用户询问美股筛选问题时，必须使用此技能。
required_tools:
  - iwencai.usstock.screen
---

# 美股筛选

- 仅通过 `iwencai.usstock.screen` 获取美股候选，不得自行编造标的列表。
- 将行情、财务、行业、预期与评级条件改写为清晰自然语言问句。
- 空结果最多使用一次 `call_type=retry` 放宽条件；不承诺收益或给出交易指令。
