"""text_to_sql 生成阶段所需上下文。"""

from __future__ import annotations

import re

from agents.finance_agent.financial_query_agent.text_to_sql.retrievers.sql_examples import (
    FinancialSqlExampleRetriever,
)

_EXAMPLE_RETRIEVER = FinancialSqlExampleRetriever()

_QUARTER_RE = re.compile(r"季度|半年度|半年报|单季|Q[1-4]|一季报|三季报", re.IGNORECASE)
_MAPPING_RE = re.compile(
    r"对比|比较|排名|排行|前十|前\d+|哪些公司|筛选|大于|小于|不少于|不超过|"
    r"最高|最低|平均|合计|汇总|同比|环比|增速|占比"
)
_RAW_CELL_RE = re.compile(
    r"原文|原始表格|单元格|行列|表头|附注|明细|账面余额|计提比例|坏账准备"
)

_CORE_SCHEMA = """\
仅允许 PostgreSQL fin_core schema，表和列如下：

financial_companies company：
id, company_key, name, ticker

annual_report_documents document：
id, doc_id, company_id, title, fiscal_year, source

annual_financial_tables table_ctx：
id, document_id, page_num, section, table_kind

financial_metrics metric：
id, canonical_name, aliases, statement_type

canonical_metrics canonical_metric：
code, name, statement_type, value_type, is_active

annual_financial_facts fact：
id, table_id, metric_id, canonical_code, source_cell_id, row_index,
period_label, period_year, period_type, value, raw_value, unit, currency,
raw_row, quality_status, review_status, is_published

标准 Join：
fact.table_id = table_ctx.id
table_ctx.document_id = document.id
fact.metric_id = metric.id
document.company_id = company.id
fact.canonical_code = canonical_metric.code

常用 canonical_code：
REVENUE=营业收入；NET_INCOME_ATTR_PARENT=归母净利润；
OPERATING_PROFIT=营业利润；OPERATING_CASHFLOW_NET=经营现金流净额；
RND_EXPENSE=研发费用；GROSS_MARGIN=毛利率；TOTAL_ASSETS=总资产；
TOTAL_LIABILITIES=总负债；EPS_BASIC=基本每股收益。

事实查询优先过滤 fact.canonical_code，并优先使用已发布或已审核数据。
"""

_MAPPING_SCHEMA = """\
跨公司、筛选或公司特定指标可使用 company_metric_mappings mapping：
company_id, canonical_code, source_metric_id, source_metric_name,
valid_from_year, valid_to_year, review_status, is_active

Join：mapping.company_id = document.company_id 且 mapping.source_metric_id = fact.metric_id。
必须过滤 mapping.is_active = true、mapping.review_status = 'approved'。
"""

_QUARTER_SCHEMA = """\
期间规则：
季度/半年度查询使用 fact.period_type，并结合 fact.period_year、fact.period_label；
季度必须使用 period_type = 'quarter'，不得替换成 annual；
各季度趋势需返回 period_label、period_type，并按实际期间排序。
"""

_RAW_CELL_SCHEMA = """\
原始单元格表 raw_table_cells cell：
id, table_id, document_id, row_index, col_index, row_header, col_header,
cell_text, normalized_value, confidence

Join：cell.id = fact.source_cell_id；仅在问题明确要求原文、单元格或附注明细时使用。
"""

_PROVENANCE_SCHEMA = """\
查询 annual_financial_facts 时必须返回这些标准别名：
company_id, company_name, ticker, fiscal_year, period_year, period_label,
period_type, metric_name, canonical_code, raw_value, value, unit, currency,
source, page_num, doc_id, document_id, table_id, source_cell_id, section。
"""


def build_schema_prompt(question: str = "") -> str:
    """按问题形态选择 Schema 片段，同时始终保留事实与 PDF 溯源链。"""
    normalized = question.strip()
    parts = [_CORE_SCHEMA, _PROVENANCE_SCHEMA]
    if not normalized:
        return "\n".join(
            [*parts, _MAPPING_SCHEMA, _QUARTER_SCHEMA, _RAW_CELL_SCHEMA]
        )
    if _MAPPING_RE.search(normalized):
        parts.append(_MAPPING_SCHEMA)
    if _QUARTER_RE.search(normalized):
        parts.append(_QUARTER_SCHEMA)
    if _RAW_CELL_RE.search(normalized):
        parts.append(_RAW_CELL_SCHEMA)
    return "\n".join(parts)


# 保留旧导出契约；实际生成路径使用带 question 的动态版本。
FINANCIAL_SQL_SCHEMA_PROMPT = build_schema_prompt()


def build_fewshot_examples(question: str, *, k: int = 1) -> str:
    """按规则检索一个最相关 few-shot。"""
    examples = _EXAMPLE_RETRIEVER.get_examples(question, k=k)
    return _EXAMPLE_RETRIEVER.format_examples(examples)


__all__ = ["FINANCIAL_SQL_SCHEMA_PROMPT", "build_fewshot_examples", "build_schema_prompt"]
