---
name: fund-screening
description: 根据基金类型、业绩、基金经理、风险、持仓、资产配置等维度筛选公募基金。返回符合条件的相关基金数据。当用户询问基金筛选问题时，必须使用此技能。
required_tools:
  - iwencai.fund.screen
---

# 基金筛选

- 仅通过 `iwencai.fund.screen` 获取基金候选和指标。
- 先确认基金类型、统计区间和风险口径，再执行筛选。
- 返回候选集及命中条件，不承诺收益；空结果不得自行放宽条件。
