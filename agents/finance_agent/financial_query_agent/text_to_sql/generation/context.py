"""text_to_sql 生成阶段所需上下文。"""

from __future__ import annotations

from agents.finance_agent.financial_query_agent.text_to_sql.retrievers.sql_examples import (
    FinancialSqlExampleRetriever,
)

_EXAMPLE_RETRIEVER = FinancialSqlExampleRetriever()

FINANCIAL_SQL_SCHEMA_PROMPT = """\
表 fin_core.financial_companies：
- id: 公司主键
- company_key: 公司规范化标识
- name: 公司名称
- ticker: 股票代码

表 fin_core.annual_report_documents：
- id: 文档主键
- doc_id: 文档唯一标识
- company_id: 对应公司主键，关联 financial_companies.id
- title: 文档标题
- fiscal_year: 财年
- source: 原始文件名或来源

表 fin_core.annual_financial_tables：
- id: 表格主键
- document_id: 对应年报文档主键，关联 annual_report_documents.id
- chunk_index: 文档内表格序号
- page_num: 页码
- section: 所在章节
- table_kind: 报表类别

表 fin_core.financial_metrics：
- id: 指标主键
- canonical_name: PDF 抽取出的 source metric 原始指标名，不等同于统一业务口径
- aliases: 原始抽取别名，可能为空
- statement_type: 原始表格或章节类型

表 fin_core.canonical_metrics：
- code: 统一指标编码，例如 REVENUE、NET_INCOME_ATTR_PARENT
- name: 统一指标名称
- statement_type: 统一指标所属报表类型
- value_type: amount / ratio / count
- description: 指标口径说明
- is_active: 是否启用

表 fin_core.canonical_metric_aliases：
- id: 别名主键
- canonical_code: 对应 canonical_metrics.code
- alias: 用户可能输入的指标词
- normalized_alias: 归一化别名
- priority: 匹配优先级
- is_active: 是否启用

表 fin_core.company_metric_mappings：
- id: 映射主键
- company_id: 公司主键
- canonical_code: 统一指标编码
- source_metric_id: 对应 financial_metrics.id
- source_metric_name: 该公司报表里的实际指标名
- valid_from_year / valid_to_year: 适用年份范围
- confidence: 映射置信度
- review_status: approved 才可作为可信映射
- is_active: 是否启用

表 fin_core.annual_financial_facts：
- id: 事实主键
- table_id: 对应财务表主键，关联 annual_financial_tables.id
- metric_id: 对应指标主键，关联 financial_metrics.id
- canonical_code: 统一指标编码；有值时优先用该字段做口径查询
- source_cell_id: 来源单元格证据，关联 raw_table_cells.id
- row_index: 原始表格行号
- period_label: 原始期间标签
- period_year: 标准化年份
- period_type: 期间类型
- value: 标准化数值
- raw_value: 原始文本值
- unit: 单位
- currency: 币种
- raw_row: 原始行文本
- quality_status: pending / passed / failed / reviewed
- review_status: unreviewed / approved / rejected
- is_published: 是否发布给 agent 精确查询

常用 canonical_code：
- 营业收入 / 营收 / 收入: REVENUE
- 归属于上市公司股东的净利润 / 归母净利润 / 净利润: NET_INCOME_ATTR_PARENT
- 营业利润 / 经营利润: OPERATING_PROFIT
- 经营活动产生的现金流量净额 / 经营现金流净额: OPERATING_CASHFLOW_NET
- 研发费用 / 研发支出: RND_EXPENSE
- 毛利率: GROSS_MARGIN
- 总资产 / 资产总计: TOTAL_ASSETS
- 总负债 / 负债合计: TOTAL_LIABILITIES
- 基本每股收益 / EPS: EPS_BASIC

表 fin_core.raw_table_cells：
- id: 单元格主键
- table_id: 表格主键
- document_id: 文档主键
- row_index / col_index: 单元格位置
- row_header / col_header: 行列头
- cell_text: 原始单元格文本
- normalized_value: 标准化数值
- confidence: 抽取置信度

推荐 Join 路径：
fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
LEFT JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = fact.canonical_code

查询约束：
- 精确财务指标查询优先使用 fact.canonical_code 或 company_metric_mappings，不要把 financial_metrics.canonical_name 当统一口径。
- 跨公司或公司特定指标查询优先使用 company_metric_mappings，并要求 mapping.is_active = true、mapping.review_status = 'approved'。
- 当数据库完成发布状态回填后，优先筛选 fact.is_published = true 或 fact.review_status = 'approved'。
- predefined 模板仅覆盖年报（period_type=annual）；text_to_sql 需自行处理季度/半年度问题。
- 季度查询使用 fact.period_type = 'quarter'，并结合 period_year、period_label（如 第一季度/Q1/三季度）过滤；不要误用 annual 条件。
- 全年各季度趋势查询保留 period_label、period_type 列，按期间标签排序展示。
"""


def build_schema_prompt() -> str:
    """集中维护 Schema 文本，避免生成与修正阶段口径不一致。"""
    return FINANCIAL_SQL_SCHEMA_PROMPT


def build_fewshot_examples(question: str, *, k: int = 3) -> str:
    """先按规则检索 few-shot，后续可无缝替换成向量检索。"""
    examples = _EXAMPLE_RETRIEVER.get_examples(question, k=k)
    return _EXAMPLE_RETRIEVER.format_examples(examples)


__all__ = ["FINANCIAL_SQL_SCHEMA_PROMPT", "build_fewshot_examples", "build_schema_prompt"]
