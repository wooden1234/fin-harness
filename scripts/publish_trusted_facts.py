"""按严格 annual amount 规则重新发布可信事实。

用法:
    python scripts/publish_trusted_facts.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.database import AsyncSessionLocal  # noqa: E402
from app.core.logger import get_logger  # noqa: E402

logger = get_logger(service="publish_trusted_facts")


RESET_SQL = """
UPDATE fin_core.annual_financial_facts
SET
  is_published = false,
  quality_status = 'pending',
  review_status = 'unreviewed',
  validation_errors = NULL,
  updated_at = now()
WHERE canonical_code IS NOT NULL
"""


PUBLISH_SQL = """
WITH candidates AS (
  SELECT
    fact.id AS fact_id,
    mapping.canonical_code,
    mapping.confidence,
    document.company_id,
    fact.period_year,
    ROW_NUMBER() OVER (
      PARTITION BY document.company_id, mapping.canonical_code, fact.period_year
      ORDER BY
        CASE
          WHEN table_ctx.section LIKE '%主要会计数据%' THEN 1
          WHEN table_ctx.section LIKE '%主要會計數據%' THEN 1
          WHEN table_ctx.section LIKE '%主要财务指标%' THEN 1
          WHEN table_ctx.section LIKE '%主要財務指標%' THEN 1
          WHEN table_ctx.section LIKE '%近三年主要会计数据%' THEN 1
          WHEN table_ctx.section LIKE '%近三年主要會計數據%' THEN 1
          WHEN table_ctx.section LIKE '%合并利润表%' THEN 2
          WHEN table_ctx.section LIKE '%合併利潤表%' THEN 2
          WHEN table_ctx.section LIKE '%合并现金流量表%' THEN 2
          WHEN table_ctx.section LIKE '%合併現金流量表%' THEN 2
          WHEN table_ctx.section LIKE '%母公司%' THEN 8
          WHEN table_ctx.section LIKE '%比較%' THEN 9
          WHEN table_ctx.section LIKE '%比较%' THEN 9
          WHEN table_ctx.section LIKE '%第四季%' THEN 10
          WHEN table_ctx.section LIKE '%第三季%' THEN 10
          ELSE 5
        END ASC,
        CASE
          WHEN metric.canonical_name IN (
            '营业收入', '營業收入', '收入',
            '归属于上市公司股东的净利润', '歸屬於上市公司股東的淨利潤', '本公司權益持有人應佔盈利',
            '经营活动产生的现金流量净额', '經營活動產生的現金流量淨額',
            '研发费用', '研發費用',
            '营业利润', '營業利潤'
          ) THEN 1
          ELSE 3
        END ASC,
        table_ctx.page_num ASC,
        fact.id ASC
    ) AS rn
  FROM fin_core.annual_financial_facts AS fact
  JOIN fin_core.annual_financial_tables AS table_ctx
    ON table_ctx.id = fact.table_id
  JOIN fin_core.annual_report_documents AS document
    ON document.id = table_ctx.document_id
  JOIN fin_core.company_metric_mappings AS mapping
    ON mapping.company_id = document.company_id
   AND mapping.source_metric_id = fact.metric_id
  JOIN fin_core.canonical_metrics AS canonical_metric
    ON canonical_metric.code = mapping.canonical_code
  JOIN fin_core.financial_metrics AS metric
    ON metric.id = fact.metric_id
  WHERE mapping.is_active = true
    AND mapping.review_status = 'approved'
    AND canonical_metric.is_active = true
    AND canonical_metric.value_type = 'amount'
    AND fact.value IS NOT NULL
    AND fact.period_year IS NOT NULL
    AND document.fiscal_year = fact.period_year
    AND fact.period_type = 'annual'
    AND COALESCE(fact.unit, '') <> '%'
    AND fact.period_label NOT LIKE '%增减%'
    AND fact.period_label NOT LIKE '%同比%'
),
picked AS (
  SELECT fact_id, canonical_code, confidence
  FROM candidates
  WHERE rn = 1
)
UPDATE fin_core.annual_financial_facts AS fact
SET
  canonical_code = picked.canonical_code,
  confidence = GREATEST(COALESCE(fact.confidence, 0), picked.confidence),
  quality_status = 'passed',
  review_status = 'approved',
  validation_errors = NULL,
  extract_version = COALESCE(NULLIF(fact.extract_version, ''), 'annual_fact_parser_v1'),
  is_published = true,
  updated_at = now()
FROM picked
WHERE fact.id = picked.fact_id
"""


REJECT_SQL = """
UPDATE fin_core.annual_financial_facts
SET
  is_published = false,
  quality_status = 'failed',
  review_status = 'rejected',
  validation_errors = 'not_publishable_as_primary_annual_amount',
  updated_at = now()
WHERE period_type IN ('change_rate', 'period_end', 'unknown')
   OR unit = '%'
   OR period_label LIKE '%增减%'
   OR period_label LIKE '%同比%'
"""


SUMMARY_SQL = """
SELECT
  count(*)::bigint AS total,
  count(*) FILTER (WHERE canonical_code IS NOT NULL)::bigint AS canonical_filled,
  count(*) FILTER (WHERE is_published = true)::bigint AS published,
  count(*) FILTER (WHERE quality_status = 'passed')::bigint AS passed,
  count(*) FILTER (WHERE review_status = 'approved')::bigint AS approved,
  count(*) FILTER (WHERE quality_status = 'failed')::bigint AS failed,
  count(*) FILTER (WHERE review_status = 'rejected')::bigint AS rejected
FROM fin_core.annual_financial_facts
"""


async def main_async() -> None:
    async with AsyncSessionLocal() as session:
        reset_result = await session.execute(text(RESET_SQL))
        publish_result = await session.execute(text(PUBLISH_SQL))
        reject_result = await session.execute(text(REJECT_SQL))
        summary = (await session.execute(text(SUMMARY_SQL))).mappings().one()
        await session.commit()

    logger.info(
        "publish completed. "
        f"reset={reset_result.rowcount}, "
        f"published={publish_result.rowcount}, "
        f"rejected={reject_result.rowcount}, "
        f"summary={dict(summary)}"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
