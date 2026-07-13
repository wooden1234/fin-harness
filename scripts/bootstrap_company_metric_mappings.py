"""根据现有 source metric 自动补公司级 canonical 映射。

用法:
    python scripts/bootstrap_company_metric_mappings.py

策略:
1. 只扫描存在 annual 数值事实的 source metric。
2. 若 source metric 经过保守归一化后与 canonical alias 完全一致，则直接写 approved。
3. 若仅存在包含关系等弱匹配，则写 pending，供人工审核。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.database import AsyncSessionLocal  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.models.annual_financial_fact import CompanyMetricMapping  # noqa: E402

logger = get_logger(service="bootstrap_company_metric_mappings")

_SPACE_RE = re.compile(r"\s+")
_LEADING_ENUM_RE = re.compile(r"^[\(（]?[一二三四五六七八九十0-9]+[\)）、\.\:]?")
_PAREN_RE = re.compile(r"[\(（][^()（）]{0,20}[\)）]")
_PUNCT_RE = re.compile(r"[:：,，、·\-\[\]【】]")

_TRADITIONAL_MAP = str.maketrans(
    {
        "營": "营",
        "業": "业",
        "歸": "归",
        "屬": "属",
        "淨": "净",
        "潤": "润",
        "經": "经",
        "現": "现",
        "額": "额",
        "發": "发",
        "費": "费",
        "幣": "币",
        "權": "权",
        "益": "益",
        "應": "应",
        "佔": "占",
        "損": "损",
        "餘": "余",
        "總": "总",
        "號": "号",
        "變": "变",
        "動": "动",
    }
)


def normalize_metric_name(value: str) -> str:
    """将 source metric 压缩到可用于高置信对齐的保守形态。"""
    text_value = (value or "").strip().translate(_TRADITIONAL_MAP)
    if not text_value:
        return ""
    text_value = _LEADING_ENUM_RE.sub("", text_value)
    text_value = _PAREN_RE.sub("", text_value)
    text_value = text_value.replace("归属于上市公司股东的", "归母")
    text_value = text_value.replace("归属于母公司股东的", "归母")
    text_value = text_value.replace("本公司权益持有人应占", "归母")
    text_value = text_value.replace("经营活动产生的", "经营")
    text_value = text_value.replace("现金流量净额", "现金流")
    text_value = text_value.replace("活動產生的", "活动产生的")
    text_value = _PUNCT_RE.sub("", text_value)
    text_value = _SPACE_RE.sub("", text_value)
    return text_value.lower()


def match_strength(source_norm: str, alias_norm: str) -> tuple[str, float] | None:
    if not source_norm or not alias_norm:
        return None
    if source_norm == alias_norm:
        return "approved", 0.97
    if source_norm.endswith(alias_norm) or alias_norm.endswith(source_norm):
        return "pending", 0.75
    if source_norm in alias_norm or alias_norm in source_norm:
        return "pending", 0.65
    return None


async def load_aliases(session) -> dict[str, list[tuple[str, str]]]:
    sql = text(
        """
        SELECT canonical_code, alias, normalized_alias
        FROM fin_core.canonical_metric_aliases
        WHERE is_active = true
        ORDER BY canonical_code, priority ASC, id ASC
        """
    )
    rows = (await session.execute(sql)).mappings().all()
    by_code: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_code.setdefault(row["canonical_code"], []).append(
            (row["alias"], normalize_metric_name(row["alias"]))
        )
    return by_code


async def load_candidate_metrics(session):
    sql = text(
        """
        SELECT
          c.id AS company_id,
          c.name AS company_name,
          m.id AS metric_id,
          m.canonical_name AS source_metric_name,
          m.statement_type,
          count(*)::bigint AS fact_rows,
          min(f.period_year) AS min_year,
          max(f.period_year) AS max_year,
          max(COALESCE(f.unit, '')) AS unit
        FROM fin_core.annual_financial_facts AS f
        JOIN fin_core.annual_financial_tables AS t
          ON t.id = f.table_id
        JOIN fin_core.annual_report_documents AS d
          ON d.id = t.document_id
        JOIN fin_core.financial_companies AS c
          ON c.id = d.company_id
        JOIN fin_core.financial_metrics AS m
          ON m.id = f.metric_id
        LEFT JOIN fin_core.company_metric_mappings AS existing
          ON existing.company_id = c.id
         AND existing.source_metric_id = m.id
         AND existing.is_active = true
        WHERE existing.id IS NULL
          AND f.period_type = 'annual'
          AND f.value IS NOT NULL
          AND COALESCE(f.unit, '') <> '%'
          AND m.canonical_name <> '合计'
          AND m.canonical_name <> '合計'
        GROUP BY c.id, c.name, m.id, m.canonical_name, m.statement_type
        ORDER BY c.name, fact_rows DESC, m.canonical_name
        """
    )
    return (await session.execute(sql)).mappings().all()


async def upsert_mapping(session, *, candidate, canonical_code: str, confidence: float, review_status: str) -> None:
    stmt = insert(CompanyMetricMapping).values(
        {
            "company_id": candidate["company_id"],
            "canonical_code": canonical_code,
            "source_metric_id": candidate["metric_id"],
            "source_metric_name": candidate["source_metric_name"],
            "statement_type": candidate["statement_type"],
            "valid_from_year": candidate["min_year"],
            "valid_to_year": candidate["max_year"],
            "priority": 50 if review_status == "approved" else 100,
            "confidence": confidence,
            "mapping_source": "bootstrap",
            "review_status": review_status,
            "is_active": True,
        }
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_company_metric_mapping_source",
        set_={
            "source_metric_name": stmt.excluded.source_metric_name,
            "statement_type": stmt.excluded.statement_type,
            "valid_from_year": stmt.excluded.valid_from_year,
            "valid_to_year": stmt.excluded.valid_to_year,
            "priority": stmt.excluded.priority,
            "confidence": stmt.excluded.confidence,
            "mapping_source": stmt.excluded.mapping_source,
            "review_status": stmt.excluded.review_status,
            "is_active": True,
        },
    )
    await session.execute(stmt)


async def main_async() -> None:
    approved = 0
    pending = 0
    async with AsyncSessionLocal() as session:
        aliases_by_code = await load_aliases(session)
        candidates = await load_candidate_metrics(session)

        for candidate in candidates:
            source_norm = normalize_metric_name(candidate["source_metric_name"])
            best: tuple[str, float, str] | None = None
            for canonical_code, aliases in aliases_by_code.items():
                for _alias, alias_norm in aliases:
                    matched = match_strength(source_norm, alias_norm)
                    if matched is None:
                        continue
                    review_status, confidence = matched
                    score = confidence
                    if best is None or score > best[1]:
                        best = (canonical_code, score, review_status)

            if best is None:
                continue

            canonical_code, confidence, review_status = best
            await upsert_mapping(
                session,
                candidate=candidate,
                canonical_code=canonical_code,
                confidence=confidence,
                review_status=review_status,
            )
            if review_status == "approved":
                approved += 1
            else:
                pending += 1

        await session.commit()

    logger.info(
        f"bootstrap completed. approved_mappings={approved}, pending_mappings={pending}"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
