---
name: research-report-search
description: 查询券商和研究机构研报，获取标题、摘要、发布时间、机构、评级、目标价及报告链接时使用。
required_tools:
  - iwencai.report.search
---

# 研报搜索

- 仅通过 `iwencai.report.search` 检索研报元数据。
- 当前阶段只回答标题、摘要、发布时间、机构名称、评级/目标价和 URL/PDF 路径；未下载或解析 PDF 时不得声称读过全文。
- 需要内容解读时，先确认已有可访问的 PDF 或正文，再交给文档解析流程；没有正文就报告限制。
