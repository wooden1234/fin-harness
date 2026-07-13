"""将当前财务 canonical 指标种子写入数据库。

用法:
    python scripts/seed_financial_canonical_metrics.py

该脚本只做 upsert，不删除人工维护的映射。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.agents.finance_agent.financial_query_agent.predefined.semantic.registry_seed import (  # noqa: E402
    CANONICAL_METRICS,
    COMPANY_OVERRIDES,
    GLOBAL_ALIASES,
)
from app.agents.finance_agent.financial_query_agent.services.entity_resolver import (  # noqa: E402
    EntityResolver,
)
from app.core.database import AsyncSessionLocal  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.models.annual_financial_fact import (  # noqa: E402
    AnnualFinancialFact,
    AnnualFinancialTable,
    AnnualReportDocument,
    CanonicalMetric,
    CanonicalMetricAlias,
    CompanyMetricMapping,
    FinancialCompany,
    FinancialMetric,
)

logger = get_logger(service="seed_financial_canonical_metrics")


def normalize_alias(value: str) -> str:
    return value.replace(" ", "").lower()


async def upsert_canonical_metrics(session) -> None:
    for metric in CANONICAL_METRICS.values():
        stmt = insert(CanonicalMetric).values(
            {
                "code": metric.code,
                "name": metric.name,
                "statement_type": None,
                "value_type": "amount",
                "default_unit": None,
                "description": metric.description,
                "is_active": True,
            }
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["code"],
            set_={
                "name": stmt.excluded.name,
                "description": stmt.excluded.description,
                "is_active": True,
                "updated_at": func.now(),
            },
        )
        await session.execute(stmt)


async def upsert_aliases(session) -> None:
    canonical_names = {
        metric.name: metric.code
        for metric in CANONICAL_METRICS.values()
    }
    alias_to_code = {**GLOBAL_ALIASES, **canonical_names}
    for alias, canonical_code in alias_to_code.items():
        stmt = insert(CanonicalMetricAlias).values(
            {
                "canonical_code": canonical_code,
                "alias": alias,
                "normalized_alias": normalize_alias(alias),
                "source": "seed",
                "priority": 10,
                "is_active": True,
            }
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_canonical_metric_alias",
            set_={
                "canonical_code": stmt.excluded.canonical_code,
                "normalized_alias": stmt.excluded.normalized_alias,
                "source": stmt.excluded.source,
                "priority": stmt.excluded.priority,
                "is_active": True,
                "updated_at": func.now(),
            },
        )
        await session.execute(stmt)


async def find_company_id(session, company_key: str) -> int | None:
    terms = {
        term.strip()
        for alias in EntityResolver.COMPANY_ALIASES.get(company_key, (company_key,))
        for term in EntityResolver.expand_company_terms(alias)
        if term.strip()
    }
    exact_terms = {term.lower() for term in terms if not term.isdigit()}
    ticker_terms = {term for term in terms if term.isdigit()}
    conditions = []
    if exact_terms:
        conditions.extend(
            [
                func.lower(FinancialCompany.name).in_(exact_terms),
                func.lower(FinancialCompany.company_key).in_(exact_terms),
            ]
        )
    if ticker_terms:
        conditions.append(FinancialCompany.ticker.in_(ticker_terms))
    if not conditions:
        return None
    stmt = select(FinancialCompany.id).where(or_(*conditions)).limit(1)
    row = (await session.execute(stmt)).first()
    return int(row.id) if row is not None else None


async def find_metric_id_for_company(
    session,
    *,
    company_id: int,
    metric_name: str,
) -> tuple[int, str, str | None] | None:
    lowered = metric_name.lower()
    base = (
        select(FinancialMetric.id, FinancialMetric.canonical_name, FinancialMetric.statement_type)
        .join(AnnualFinancialFact, AnnualFinancialFact.metric_id == FinancialMetric.id)
        .join(AnnualFinancialTable, AnnualFinancialTable.id == AnnualFinancialFact.table_id)
        .join(AnnualReportDocument, AnnualReportDocument.id == AnnualFinancialTable.document_id)
        .where(AnnualReportDocument.company_id == company_id)
    )

    exact = (await session.execute(
        base.where(func.lower(FinancialMetric.canonical_name) == lowered).limit(1)
    )).first()
    if exact is not None:
        return int(exact.id), exact.canonical_name, exact.statement_type

    fuzzy = (await session.execute(
        base.where(func.lower(FinancialMetric.canonical_name).like(f"%{lowered}%")).limit(1)
    )).first()
    if fuzzy is not None:
        return int(fuzzy.id), fuzzy.canonical_name, fuzzy.statement_type
    return None


async def upsert_company_mappings(session) -> int:
    created = 0
    for company_key, mappings in COMPANY_OVERRIDES.items():
        company_id = await find_company_id(session, company_key)
        if company_id is None:
            logger.warning(f"跳过公司映射，未找到公司: {company_key}")
            continue

        for canonical_code, source_metric_names in mappings.items():
            for source_metric_name in source_metric_names:
                resolved = await find_metric_id_for_company(
                    session,
                    company_id=company_id,
                    metric_name=source_metric_name,
                )
                if resolved is None:
                    logger.warning(
                        f"跳过指标映射，未找到 source metric: "
                        f"company={company_key}, canonical={canonical_code}, metric={source_metric_name}"
                    )
                    continue

                metric_id, metric_name, statement_type = resolved
                stmt = insert(CompanyMetricMapping).values(
                    {
                        "company_id": company_id,
                        "canonical_code": canonical_code,
                        "source_metric_id": metric_id,
                        "source_metric_name": metric_name,
                        "statement_type": statement_type,
                        "priority": 10,
                        "confidence": 0.98,
                        "mapping_source": "seed",
                        "review_status": "approved",
                        "is_active": True,
                    }
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_company_metric_mapping_source",
                    set_={
                        "source_metric_name": stmt.excluded.source_metric_name,
                        "statement_type": stmt.excluded.statement_type,
                        "priority": stmt.excluded.priority,
                        "confidence": stmt.excluded.confidence,
                        "mapping_source": stmt.excluded.mapping_source,
                        "review_status": stmt.excluded.review_status,
                        "is_active": True,
                        "updated_at": func.now(),
                    },
                )
                await session.execute(stmt)
                created += 1
    return created


async def main_async() -> None:
    async with AsyncSessionLocal() as session:
        await upsert_canonical_metrics(session)
        await upsert_aliases(session)
        mapping_count = await upsert_company_mappings(session)
        await session.commit()
    logger.info(f"Seed completed. company_metric_mappings={mapping_count}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
