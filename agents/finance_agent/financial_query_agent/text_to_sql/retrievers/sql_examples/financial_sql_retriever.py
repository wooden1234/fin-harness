"""财务 SQL few-shot 示例检索器。"""

from __future__ import annotations

import re

from agents.finance_agent.financial_query_agent.text_to_sql.retrievers.sql_examples.base import (
    BaseSqlExampleRetriever,
    SqlExample,
)


class FinancialSqlExampleRetriever(BaseSqlExampleRetriever):
    """按问题特征选择最相近的财务 SQL 示例。"""

    _CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
        "单指标查数": ("多少", "是多少", "查询", "营收", "营业收入", "净利润"),
        "同比对比": ("同比", "对比", "比较", "增长", "变化"),
        "趋势分析": ("趋势", "历年", "近三年", "近五年", "近5年"),
        "多公司对比": ("和", "与", "对比", "比较", "分别"),
        "排名查询": ("排名", "前十", "最高", "最低", "top"),
        "季度查询": (
            "季度",
            "季报",
            "一季",
            "二季",
            "三季",
            "四季",
            "半年度",
            "半年",
            "q1",
            "q2",
            "q3",
            "q4",
        ),
    }

    _ALL_EXAMPLES: dict[str, tuple[SqlExample, ...]] = {
        "单指标查数": (
            SqlExample(
                category="单指标查数",
                question="宁德时代 2024 年营业收入是多少",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  document.fiscal_year AS fiscal_year,
  fact.period_year AS period_year,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(fact.currency, '') AS currency,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name = :company_name
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) IN :years
ORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC
LIMIT :limit
""",
                notes=["单公司单指标优先使用 canonical_code 和公司级映射", "结果列别名保持统一"],
            ),
        ),
        "同比对比": (
            SqlExample(
                category="同比对比",
                question="比亚迪 2023 和 2024 年净利润对比",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  COALESCE(fact.period_year, document.fiscal_year) AS period_year,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(fact.currency, '') AS currency,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name = :company_name
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) IN :years
ORDER BY period_year ASC
LIMIT :limit
""",
                notes=["年份对比保留 period_year 方便上游汇总", "指标口径使用 canonical_code"],
            ),
        ),
        "趋势分析": (
            SqlExample(
                category="趋势分析",
                question="茅台近 5 年毛利率变化",
                sql="""
SELECT
  recent.company_id,
  recent.company_name,
  recent.period_year,
  recent.period_type,
  recent.metric_name,
  recent.raw_value,
  recent.unit,
  recent.source,
  recent.page_num,
  recent.doc_id
FROM (
  SELECT
    company.id AS company_id,
    company.name AS company_name,
    COALESCE(fact.period_year, document.fiscal_year) AS period_year,
    fact.period_type AS period_type,
    canonical_metric.name AS metric_name,
    COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
    COALESCE(fact.unit, '') AS unit,
    COALESCE(document.source, '') AS source,
    table_ctx.page_num AS page_num,
    COALESCE(document.doc_id, '') AS doc_id,
    document.id AS document_id,
    table_ctx.id AS table_id,
    fact.source_cell_id AS source_cell_id,
    COALESCE(table_ctx.section, '') AS section
  FROM fin_core.annual_financial_facts AS fact
  JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
  JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
  JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
  JOIN fin_core.company_metric_mappings AS mapping
    ON mapping.company_id = document.company_id
   AND mapping.source_metric_id = fact.metric_id
  JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
  LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
  WHERE company.name = :company_name
    AND mapping.canonical_code = :canonical_code
    AND mapping.is_active = true
    AND mapping.review_status = 'approved'
    AND canonical_metric.is_active = true
    AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
    AND (fact.is_published = true OR fact.review_status = 'approved')
  ORDER BY COALESCE(fact.period_year, document.fiscal_year) DESC
  LIMIT :limit
) AS recent
ORDER BY period_year ASC
""",
                notes=["近 N 年趋势不需要用户给具体年份", "先取最近 N 个年度，再按年份升序展示"],
            ),
        ),
        "多公司对比": (
            SqlExample(
                category="多公司对比",
                question="宁德时代和比亚迪 2024 年营业收入对比",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  COALESCE(fact.period_year, document.fiscal_year) AS period_year,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name IN :company_names
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) IN :years
ORDER BY company.name ASC, period_year DESC
LIMIT :limit
""",
                notes=["多公司比较使用 IN :company_names", "同一指标对比使用 canonical_code"],
            ),
        ),
        "排名查询": (
            SqlExample(
                category="排名查询",
                question="2024 年营业收入最高的前十家公司",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  COALESCE(fact.period_year, document.fiscal_year) AS period_year,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(CAST(fact.value AS TEXT), fact.raw_value, '') AS value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND (fact.period_type = 'annual' OR fact.period_type IS NULL)
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) = :year
ORDER BY fact.value DESC NULLS LAST
LIMIT :limit
""",
                notes=["排名查询优先按数值列排序", "排名指标使用 canonical_code，仍需输出展示别名"],
            ),
        ),
        "季度查询": (
            SqlExample(
                category="季度查询",
                question="宁德时代 2024 年一季度营业收入是多少",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  company.ticker AS ticker,
  document.fiscal_year AS fiscal_year,
  fact.period_year AS period_year,
  fact.period_label AS period_label,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(fact.currency, '') AS currency,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name = :company_name
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND fact.period_type = 'quarter'
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) = :year
  AND (
    fact.period_label ILIKE :quarter_label_pattern
    OR fact.period_label ILIKE :quarter_label_pattern_alt
  )
ORDER BY fact.period_label ASC
LIMIT :limit
""",
                notes=[
                    "季度问题必须筛选 fact.period_type = 'quarter'，不要用 annual 条件替代",
                    "单季查询结合 period_label（第一季度/Q1 等）与 period_year",
                ],
            ),
            SqlExample(
                category="季度查询",
                question="比亚迪 2024 年各季度净利润",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  COALESCE(fact.period_year, document.fiscal_year) AS period_year,
  fact.period_label AS period_label,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name = :company_name
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND fact.period_type = 'quarter'
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) = :year
ORDER BY fact.period_label ASC
LIMIT :limit
""",
                notes=[
                    "全年各季度查询保留 period_label 与 period_type 列",
                    "按 period_label 升序展示一季度到四季度",
                ],
            ),
            SqlExample(
                category="季度查询",
                question="腾讯 2023 年 Q3 营业收入",
                sql="""
SELECT
  company.id AS company_id,
  company.name AS company_name,
  COALESCE(fact.period_year, document.fiscal_year) AS period_year,
  fact.period_label AS period_label,
  fact.period_type AS period_type,
  canonical_metric.name AS metric_name,
  COALESCE(fact.raw_value, CAST(fact.value AS TEXT), '') AS raw_value,
  COALESCE(fact.unit, '') AS unit,
  COALESCE(document.source, '') AS source,
  table_ctx.page_num AS page_num,
  COALESCE(document.doc_id, '') AS doc_id,
  document.id AS document_id,
  table_ctx.id AS table_id,
  fact.source_cell_id AS source_cell_id,
  COALESCE(table_ctx.section, '') AS section
FROM fin_core.annual_financial_facts AS fact
JOIN fin_core.annual_financial_tables AS table_ctx ON table_ctx.id = fact.table_id
JOIN fin_core.annual_report_documents AS document ON document.id = table_ctx.document_id
JOIN fin_core.financial_metrics AS metric ON metric.id = fact.metric_id
JOIN fin_core.company_metric_mappings AS mapping
  ON mapping.company_id = document.company_id
 AND mapping.source_metric_id = fact.metric_id
JOIN fin_core.canonical_metrics AS canonical_metric ON canonical_metric.code = mapping.canonical_code
LEFT JOIN fin_core.financial_companies AS company ON company.id = document.company_id
WHERE company.name = :company_name
  AND mapping.canonical_code = :canonical_code
  AND mapping.is_active = true
  AND mapping.review_status = 'approved'
  AND canonical_metric.is_active = true
  AND fact.period_type = 'quarter'
  AND (fact.is_published = true OR fact.review_status = 'approved')
  AND COALESCE(fact.period_year, document.fiscal_year) = :year
  AND (
    fact.period_label ILIKE :quarter_label_pattern
    OR fact.period_label ILIKE '%三%'
    OR fact.period_label ILIKE '%3%'
  )
ORDER BY fact.period_label ASC
LIMIT :limit
""",
                notes=["Q1-Q4 缩写需映射到 period_label 模糊匹配", "季度查询不要套用 predefined 的年报过滤"],
            ),
        ),
    }

    def get_examples(self, query: str, *, k: int = 3) -> list[SqlExample]:
        normalized = query.lower()
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", normalized))
        scored: list[tuple[int, SqlExample]] = []

        for category, examples in self._ALL_EXAMPLES.items():
            category_score = self._category_score(category, normalized, tokens)
            for example in examples:
                example_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", example.question.lower()))
                overlap_score = len(tokens & example_tokens)
                total_score = category_score * 10 + overlap_score
                scored.append((total_score, example))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[SqlExample] = []
        seen_questions: set[str] = set()
        for score, example in scored:
            if score <= 0:
                continue
            if example.question in seen_questions:
                continue
            selected.append(example)
            seen_questions.add(example.question)
            if len(selected) >= k:
                break

        if selected:
            return selected

        fallback_examples = [examples[0] for examples in self._ALL_EXAMPLES.values() if examples]
        return fallback_examples[:k]

    def _category_score(self, category: str, query: str, tokens: set[str]) -> int:
        score = 0
        for keyword in self._CATEGORY_KEYWORDS.get(category, ()):
            if keyword in query:
                score += 1

        if category == "多公司对比" and len(self._extract_companies(query)) >= 2:
            score += 2
        if category in {"同比对比", "趋势分析"} and len(self._extract_years(query)) >= 2:
            score += 1
        if category == "排名查询" and ("前" in tokens or "top" in tokens):
            score += 1
        if category == "季度查询":
            if re.search(r"q[1-4]", query, re.IGNORECASE):
                score += 2
            if any(token in query for token in ("一季度", "二季度", "三季度", "四季度", "各季度", "分季度")):
                score += 2
        if category == "单指标查数" and self._looks_like_quarter_query(query):
            score -= 2
        return score

    @staticmethod
    def _looks_like_quarter_query(query: str) -> bool:
        lowered = query.lower()
        if re.search(r"q[1-4]", lowered):
            return True
        return any(
            token in query
            for token in ("季度", "季报", "半年度", "半年", "一季", "二季", "三季", "四季")
        )

    @staticmethod
    def _extract_years(query: str) -> list[str]:
        return re.findall(r"(20\d{2})", query)

    @staticmethod
    def _extract_companies(query: str) -> list[str]:
        parts = re.split(r"[和与、,，]", query)
        return [part.strip() for part in parts if part.strip()]


__all__ = ["FinancialSqlExampleRetriever"]
