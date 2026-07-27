---
name: announcement-search
description: 查询 A 股、港股、基金和 ETF 公告、财报、分红、回购、重组等披露信息时使用。
required_tools:
  - iwencai.announcement.search
---

# 公告搜索

- 仅通过 `iwencai.announcement.search` 检索公告，不自行构造 HTTP 请求。
- 输出标题、摘要、发布时间、标的、公告类型和 URL/PDF 路径（以工具返回为准）。
- 搜索无结果时明确说明；不得把新闻或模型推测当作公告事实。
