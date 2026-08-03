---
name: financial-research
description: 基于官方、行情、研究和知识库来源完成金融主题研究。用于行业前景、公司基本面、周期位置、风险和投资价值分析。
allowed-tools: query_iwencai compare_entities_with_iwencai query_iwencai_industry query_iwencai_market query_iwencai_rating search_iwencai_report search_web catalog_pdf_knowledge_tool search_pdf_knowledge_tool search_faq_knowledge_tool run_calculation
---

# 金融研究

1. 把问题拆成事实、计算、推断和反方证据待办项。
2. 财务事实优先用 `query_iwencai`（单实体）或 `compare_entities_with_iwencai`（2 个以上实体同指标，一次并发查询）；行情用 `query_iwencai_market`/`query_iwencai_industry`；本地已收录 PDF/FAQ 用 `catalog_pdf_knowledge_tool`/`search_pdf_knowledge_tool`/`search_faq_knowledge_tool`。
3. 开放判断至少取得两个独立来源族，`run_calculation` 不算独立来源。
4. 区分已披露事实、机构预期（`query_iwencai_rating`/`search_iwencai_report`）和分析推断，不用模型记忆补数值。
5. 证据不足时保留已核验事实，并明确缺失来源和未决问题。
6. 投资动作问题只讨论条件、风险和观察指标，不给个性化交易指令。

