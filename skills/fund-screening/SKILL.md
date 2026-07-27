---
name: fund-screening
description: 按基金类型、业绩、基金经理、风险、持仓或资产配置条件筛选基金时使用。
required_tools:
  - iwencai.fund.screen
---

# 基金筛选

- 仅通过 `iwencai.fund.screen` 获取基金候选和指标。
- 先确认基金类型、统计区间和风险口径，再执行筛选。
- 返回候选集及命中条件，不承诺收益；空结果不得自行放宽条件。
