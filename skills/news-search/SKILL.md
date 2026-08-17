---
name: news-search
description: 搜索财经新闻、政策动态、行业革新和企业业务进展。当用户询问最新财经事件、政策、板块上涨原因、公司经营进展时使用；不替代通用网页搜索。
required_tools:
  - iwencai.news.search
---

# 财经新闻搜索

- 仅通过 `iwencai.news.search` 检索同花顺财经资讯，不自行构造 HTTP 请求。
- 适合：最新财经事件、政策动态、行业革新、公司业务进展、板块上涨的新闻驱动。
- **不要**用本工具查询板块涨幅排名或财务数字；排名走 `iwencai.industry.query`，报表数字走 finance-query / local-financial-facts。
- **不要**替代 `web.search`：非财经网页、境外英文来源、问财库没有的站点仍用网页搜索。
- 输出标题、摘要、发布时间、来源和 URL（以工具返回为准）；注明数据来源为同花顺问财。
- 无结果时明确说明，不得用模型记忆编造新闻。
