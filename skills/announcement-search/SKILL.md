---
name: announcement-search
description: 支持A股、港股、基金、ETF等金融标的公告的查询，同时公告类型包括不限于定期财务报告、分红派息、回购增持、资产重组等等。
required_tools:
  - iwencai.announcement.search
---

# 公告搜索

- 仅通过 `iwencai.announcement.search` 检索公告，不自行构造 HTTP 请求。
- 输出标题、摘要、发布时间、标的、公告类型和 URL/PDF 路径（以工具返回为准）。
- 搜索无结果时明确说明；不得把新闻或模型推测当作公告事实。
