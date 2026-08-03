---
name: rumor-verification
description: 核验公司、行业和市场传闻并区分确认、否认与未证实信息。用于并购、合作、产品、监管和突发市场消息核查。
allowed-tools: search_iwencai_announcement search_web query_iwencai query_iwencai_market run_calculation
---

# 传闻核验

1. 将传闻拆成主体、事件、时间和可证伪陈述。
2. 先用 `search_iwencai_announcement` 检索公司公告、监管披露或官方回应，再用 `search_web` 查独立新闻来源。
3. 记录支持证据和反方证据，不以转述数量代替独立来源。
4. 只有来源和时间明确时才标记为已确认；否则标记为未证实。
5. 评估行业或股价影响时再用 `query_iwencai_market`/`query_iwencai` 补行情或研究来源。
6. 不把市场传闻转化为确定收益、目标价或交易建议。
