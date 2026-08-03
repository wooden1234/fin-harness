---
name: earnings-comparison
description: 比较两家或多家公司已披露财报、业务增速、市场预期和前瞻指引。用于最新财季对比、业绩亮点比较和超预期分析。
allowed-tools: compare_entities_with_iwencai query_iwencai query_iwencai_rating search_iwencai_announcement search_web run_calculation catalog_pdf_knowledge_tool search_pdf_knowledge_tool
---

# 财报对比

1. 为每个实体建立待办项，先取得同一完整财季的官方披露。
   数据未取得或检索零命中时不得把该待办标为 completed，应保留缺口并切换来源。
2. **两个及以上实体、同一指标的数值对比，优先调用 `compare_entities_with_iwencai`**（一次并发查询所有实体，比逐个调用 `query_iwencai` 更省配额）：
   - `entities` 传全部待比较公司/指数名称（2-6 个）。
   - `query` 写共用指标短语，如「近两个完整财年营业收入 归母净利润 销售毛利率」，不要把公司名写进 query（工具会自动拼接）。
   - 返回的 `calibre.multi_currency` / `calibre.multi_period` 为 true 时，成稿必须在 gaps 中说明口径差异，但仍要给出对照，不得因此拒答（参考第 4/5 条）。
   - 若某实体在 `failed_entities` 中，改用 `query_iwencai` 单独重试该实体，不必重查已成功的实体。
3. 逐实体核对财季、发布日期、营收或利润及分业务增速。
4. 涉及“超预期”时，再用 `query_iwencai_rating` 取得一致预期、评级或研报来源。
5. 财季或财年起止不一致时仍给出对照：分列各方已披露数值与同比/增速，并在 gaps 中写明「非同一会计期间」；仅当一方完全未披露时才标注该侧缺口，不得整题拒绝对比。
6. 币种不一致时：有汇率证据则换算后横比并注明汇率日期；否则原币分列、突出增速对比，绝对金额注明币种，不得只回答无法对比。
7. 对冲突数字优先采用官方披露，保留无法统一口径的缺口。
8. 每条事实只引用工具返回的真实 Evidence ID。
9. 本地事实表仅用于白名单窄查；PDF 仅用于目录已收录文档。收到不可重试信号后，优先切换问财、公告/研报或 Web，不得在同一窄库中反复改写查询。
