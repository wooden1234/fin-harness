---
name: institution-rating
description: 查询券商研报评级、业绩预测、ESG、信用评级、目标价和机构观点时使用。
required_tools:
  - iwencai.rating.query
---

# 机构评级查询

- 仅通过 `iwencai.rating.query` 获取机构评级和目标价事实。
- 区分评级机构、评级日期、评级方向和目标价，避免把历史评级当作当前评级。
- 只汇总工具返回的内容，不将评级解释为确定收益或投资建议。
