"""Claim—Evidence 映射、证据质量评估与受约束综合。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.llm import get_router_llm
from agents.structured_output import ainvoke_json_output
from agents.orchestrator.contracts import (
    AgentResult,
    AnswerStatement,
    Claim,
    ClaimEvidenceLink,
    ConstrainedAnswer,
    Evidence,
    EvidenceAssessment,
    EvidenceConflict,
    RequestProfile,
)
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="orchestrator_evidence_quality")

_GRADE_SCORE = {"A": 1.0, "B": 0.85, "C": 0.7, "D": 0.5, "E": 0.2}
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CLAIM_EXTRACTION_JSON_CONTRACT = """仅输出合法 JSON 对象，不得输出 Markdown、代码块或额外说明。
JSON 输出格式：
{
  "claims": [
    {
      "task_id": "输入中的 task_id",
      "text": "可独立核验的陈述",
      "claim_type": "fact",
      "evidence_ids": ["输入中直接支撑该陈述的 evidence_id"]
    }
  ]
}
"""
_CONSTRAINED_SYNTHESIS_JSON_CONTRACT = """仅输出合法 JSON 对象，不得输出 Markdown、代码块或额外说明。
JSON 输出格式：
{
  "statements": [
    {
      "text": "受证据支持的中文陈述",
      "claim_ids": ["对应的 claim_id"],
      "evidence_ids": ["对应的 evidence_id"],
      "statement_type": "fact",
      "confidence": 0.8
    }
  ],
  "unresolved_claim_ids": [],
  "caveats": []
}
"""


class _ExtractedClaim(BaseModel):
    """非结构化结果中抽取的一条 Claim。"""

    task_id: str
    text: str
    subject: str = ""
    predicate: str = ""
    value: Any = None
    unit: str = ""
    currency: str = ""
    period: str | None = None
    as_of: str | None = None
    claim_type: str = "fact"
    importance: str = "major"
    evidence_ids: list[str] = Field(default_factory=list)


class _ClaimExtractionOutput(BaseModel):
    """非结构化 Claim 抽取的结构化输出。"""

    claims: list[_ExtractedClaim] = Field(default_factory=list)


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha1(raw).hexdigest()[:16]}"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(normalized[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def source_grade(evidence: Evidence) -> str:
    """根据受控来源字段计算来源等级，不信任模型自报等级。"""
    fingerprint = " ".join(
        [
            evidence.source_type,
            evidence.provider,
            evidence.title,
            evidence.url or "",
            str(evidence.metadata.get("document_type") or ""),
        ]
    ).lower()
    if any(
        marker in fingerprint
        for marker in (
            "announcement",
            "exchange",
            "official",
            "cninfo",
            "sec.gov",
            "监管",
            "交易所",
            "公告",
        )
    ):
        return "A"
    if any(
        marker in fingerprint
        for marker in (
            "financial_db",
            "financial_fact",
            "financial_query",
            "market.query",
            "industry.query",
            "index.query",
            "fund.screen",
            "iwencai.screen",
            "faq",
        )
    ):
        return "B"
    if any(
        marker in fingerprint
        for marker in ("report", "rating", "研报", "评级", "pdf")
    ):
        return "C"
    if any(marker in fingerprint for marker in ("web", "news", "research")):
        return "D"
    return "E"


def _freshness_limit_days(evidence: Evidence) -> int:
    fingerprint = f"{evidence.source_type} {evidence.title}".lower()
    if any(marker in fingerprint for marker in ("market", "screen", "行情", "股价")):
        return 2
    if any(marker in fingerprint for marker in ("report", "rating", "研报", "评级")):
        return 180
    if any(marker in fingerprint for marker in ("financial", "财务", "公告")):
        return 550
    return 30


def assess_evidence(
    evidence: Evidence,
    *,
    freshness_required: bool,
    now: datetime | None = None,
) -> EvidenceAssessment:
    """评估来源权威性、完整性和请求相关的时效性。"""
    grade = source_grade(evidence)
    provenance_fields = [
        evidence.provider,
        evidence.title,
        evidence.url or "",
        str(evidence.metadata.get("document_id") or ""),
        str(evidence.metadata.get("table_id") or ""),
    ]
    completeness = min(
        1.0,
        (
            0.4
            + (0.2 if evidence.content else 0.0)
            + (0.2 if any(provenance_fields) else 0.0)
            + (0.2 if evidence.source_type else 0.0)
        ),
    )
    rejection_reasons: list[str] = []
    if grade == "E":
        rejection_reasons.append("source_grade_too_low")
    if not evidence.content and not evidence.metadata.get("structured_data"):
        rejection_reasons.append("evidence_content_missing")
    if completeness < 0.6:
        rejection_reasons.append("provenance_incomplete")

    stale = False
    freshness_score = 1.0
    if freshness_required:
        timestamp = _parse_datetime(
            evidence.published_at
            or evidence.observed_at
            or str(evidence.metadata.get("as_of") or "")
        )
        if timestamp is None:
            stale = True
            freshness_score = 0.0
            rejection_reasons.append("freshness_timestamp_missing")
        else:
            current = now or datetime.now(timezone.utc)
            age_days = max(0.0, (current - timestamp).total_seconds() / 86400)
            limit_days = _freshness_limit_days(evidence)
            stale = age_days > limit_days
            freshness_score = max(0.0, 1.0 - age_days / max(limit_days, 1))
            if stale:
                rejection_reasons.append(
                    f"evidence_stale:{age_days:.0f}>{limit_days}"
                )

    return EvidenceAssessment(
        evidence_id=evidence.evidence_id,
        source_grade=grade,
        authority_score=_GRADE_SCORE[grade],
        freshness_score=freshness_score,
        completeness_score=completeness,
        usable=not rejection_reasons,
        stale=stale,
        rejection_reasons=rejection_reasons,
    )


def _claim_from_mapping(
    raw: dict[str, Any],
    *,
    task_id: str,
    index: int,
) -> tuple[Claim, list[str]]:
    text = str(raw.get("text") or raw.get("claim") or "").strip()
    value = raw.get("value")
    if not text:
        text = f"{raw.get('predicate') or '结论'}：{value}"
    claim_id = str(raw.get("claim_id") or "") or _stable_id(
        "claim",
        task_id,
        str(index),
        text,
    )
    claim = Claim(
        claim_id=claim_id,
        task_id=task_id,
        text=text,
        subject=str(raw.get("subject") or ""),
        predicate=str(raw.get("predicate") or ""),
        value=value,
        unit=str(raw.get("unit") or ""),
        currency=str(raw.get("currency") or ""),
        period=str(raw["period"]) if raw.get("period") else None,
        as_of=str(raw["as_of"]) if raw.get("as_of") else None,
        claim_type=(
            raw.get("claim_type")
            if raw.get("claim_type") in {"fact", "calculation", "inference", "opinion"}
            else "fact"
        ),
        importance=(
            raw.get("importance")
            if raw.get("importance") in {"critical", "major", "minor"}
            else "major"
        ),
        metadata=dict(raw.get("metadata") or {}),
    )
    return claim, [str(item) for item in raw.get("evidence_ids") or []]


def _deterministic_claims(
    results: Iterable[AgentResult],
) -> tuple[list[Claim], list[ClaimEvidenceLink], list[AgentResult]]:
    claims: list[Claim] = []
    links: list[ClaimEvidenceLink] = []
    unstructured: list[AgentResult] = []
    for result in results:
        result_claims = result.structured_data.get("claims")
        if isinstance(result_claims, list) and result_claims:
            for index, raw in enumerate(result_claims):
                if not isinstance(raw, dict):
                    continue
                claim, evidence_ids = _claim_from_mapping(
                    raw,
                    task_id=result.task_id,
                    index=index,
                )
                claims.append(claim)
                links.extend(
                    ClaimEvidenceLink(
                        claim_id=claim.claim_id,
                        evidence_id=evidence_id,
                    )
                    for evidence_id in evidence_ids
                )
            continue

        metadata_claims = [
            item
            for item in result.evidence
            if item.metadata.get("claim_key")
        ]
        if metadata_claims:
            for index, evidence in enumerate(metadata_claims):
                key = str(evidence.metadata["claim_key"])
                value = evidence.metadata.get("claim_value")
                claim = Claim(
                    claim_id=_stable_id(
                        "claim",
                        result.task_id,
                        key,
                        str(value),
                    ),
                    task_id=result.task_id,
                    text=str(evidence.metadata.get("claim_text") or f"{key}：{value}"),
                    predicate=key,
                    value=value,
                    period=(
                        str(evidence.metadata["period"])
                        if evidence.metadata.get("period")
                        else None
                    ),
                    unit=str(evidence.metadata.get("unit") or ""),
                    metadata={"claim_key": key, "source": "evidence_metadata"},
                )
                claims.append(claim)
                links.append(
                    ClaimEvidenceLink(
                        claim_id=claim.claim_id,
                        evidence_id=evidence.evidence_id,
                    )
                )
            continue
        if result.answer:
            unstructured.append(result)
    return claims, links, unstructured


async def _extract_unstructured_claims(
    results: list[AgentResult],
    *,
    config: RunnableConfig | None,
) -> tuple[list[Claim], list[ClaimEvidenceLink]]:
    if not results or not settings.DEEPSEEK_API_KEY:
        return [], []
    allowed_evidence = {
        item.evidence_id
        for result in results
        for item in result.evidence
    }
    payload = [
        {
            "task_id": result.task_id,
            "answer": result.answer[:4000],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "content": item.content[:1500],
                }
                for item in result.evidence
            ],
        }
        for result in results
    ]
    prompt = (
        "从候选答案中提取可独立核验的 Claim。只能关联输入中存在且直接支撑该 "
        "Claim 的 evidence_id；没有证据的 Claim 保留空 evidence_ids。"
        "区分 fact、calculation、inference、opinion，不得创造新事实。\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )
    try:
        output = await ainvoke_json_output(
            get_router_llm(),
            _ClaimExtractionOutput,
            [
                (
                    "system",
                    "你是金融证据工程师，只进行 Claim 抽取和证据映射。"
                    f"\n{_CLAIM_EXTRACTION_JSON_CONTRACT}",
                ),
                ("human", prompt),
            ],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim extraction fallback: {}", type(exc).__name__)
        return [], []

    claims: list[Claim] = []
    links: list[ClaimEvidenceLink] = []
    known_tasks = {item.task_id for item in results}
    for index, raw in enumerate(output.claims):
        if raw.task_id not in known_tasks or not raw.text.strip():
            continue
        claim, evidence_ids = _claim_from_mapping(
            raw.model_dump(),
            task_id=raw.task_id,
            index=index,
        )
        claims.append(claim)
        links.extend(
            ClaimEvidenceLink(
                claim_id=claim.claim_id,
                evidence_id=evidence_id,
            )
            for evidence_id in evidence_ids
            if evidence_id in allowed_evidence
        )
    return claims, links


def _fallback_claims(
    results: Iterable[AgentResult],
) -> tuple[list[Claim], list[ClaimEvidenceLink]]:
    """模型不可用时把整个 Agent 结论作为保守的粗粒度 Claim。"""
    claims: list[Claim] = []
    links: list[ClaimEvidenceLink] = []
    for result in results:
        if not result.answer:
            continue
        claim = Claim(
            claim_id=_stable_id("claim", result.task_id, result.answer),
            task_id=result.task_id,
            text=result.answer,
            claim_type="fact" if result.evidence else "opinion",
            metadata={"coarse_fallback": True},
        )
        claims.append(claim)
        links.extend(
            ClaimEvidenceLink(
                claim_id=claim.claim_id,
                evidence_id=evidence.evidence_id,
                strength=0.5,
                reason="粗粒度降级映射",
            )
            for evidence in result.evidence
        )
    return claims, links


async def build_claim_evidence_map(
    results: list[AgentResult],
    *,
    config: RunnableConfig | None = None,
    allow_llm: bool = True,
) -> tuple[list[Claim], list[ClaimEvidenceLink]]:
    """优先读取结构化 Claim，再按需抽取，最后使用保守映射。"""
    claims, links, unstructured = _deterministic_claims(results)
    extracted: list[Claim] = []
    extracted_links: list[ClaimEvidenceLink] = []
    if allow_llm:
        extracted, extracted_links = await _extract_unstructured_claims(
            unstructured,
            config=config,
        )
    extracted_tasks = {item.task_id for item in extracted}
    fallback, fallback_links = _fallback_claims(
        item for item in unstructured if item.task_id not in extracted_tasks
    )
    return (
        [*claims, *extracted, *fallback],
        [*links, *extracted_links, *fallback_links],
    )


def _claim_key(claim: Claim) -> str:
    explicit = str(claim.metadata.get("claim_key") or "").strip().lower()
    if explicit:
        return explicit
    unit = claim.unit.strip().lower()
    if any(marker in unit for marker in ("元", "万", "亿")) or claim.currency:
        unit_dimension = "money"
    elif "%" in unit or "percent" in unit:
        unit_dimension = "percent"
    else:
        unit_dimension = unit
    fields = [
        claim.subject,
        claim.predicate,
        claim.period or "",
        unit_dimension,
        claim.currency,
    ]
    if claim.subject or claim.predicate:
        return "|".join(str(item).strip().lower() for item in fields)
    return claim.claim_id


def _canonical_value(claim: Claim) -> str:
    value = claim.value
    if value is None:
        return ""
    text = str(value).strip()
    match = _NUMBER_PATTERN.search(text)
    if not match:
        return text.lower()
    try:
        number = Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return text.lower()
    unit_text = f"{claim.unit} {text}"
    if "亿" in unit_text:
        number *= Decimal("100000000")
    elif "万" in unit_text:
        number *= Decimal("10000")
    return format(number.normalize(), "f")


def detect_claim_conflicts(
    claims: list[Claim],
    links: list[ClaimEvidenceLink],
) -> list[EvidenceConflict]:
    """在实体、指标、报告期和单位一致时比较规范化值。"""
    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        if claim.value is not None:
            groups.setdefault(_claim_key(claim), []).append(claim)
    evidence_by_claim: dict[str, list[str]] = {}
    for link in links:
        evidence_by_claim.setdefault(link.claim_id, []).append(link.evidence_id)

    conflicts: list[EvidenceConflict] = []
    for key, group in groups.items():
        values = {_canonical_value(item) for item in group}
        values.discard("")
        if len(values) <= 1:
            continue
        as_of_values = {item.as_of for item in group if item.as_of}
        temporal = len(as_of_values) > 1
        preferred_claim_id = None
        if temporal:
            preferred_claim_id = max(
                group,
                key=lambda item: _parse_datetime(item.as_of)
                or datetime.min.replace(tzinfo=timezone.utc),
            ).claim_id
        conflicts.append(
            EvidenceConflict(
                conflict_id=_stable_id("conflict", key, *sorted(values)),
                claim_key=key,
                conflict_type="temporal_update" if temporal else "hard_conflict",
                claim_ids=[item.claim_id for item in group],
                evidence_ids=list(
                    dict.fromkeys(
                        evidence_id
                        for item in group
                        for evidence_id in evidence_by_claim.get(item.claim_id, [])
                    )
                ),
                values=sorted(values),
                resolved=temporal,
                preferred_claim_id=preferred_claim_id,
                resolution=(
                    "不同数据时点，综合时仅保留较新的 as_of"
                    if temporal
                    else ""
                ),
            )
        )
    return conflicts


def supported_claim_ids(
    claims: list[Claim],
    links: list[ClaimEvidenceLink],
    assessments: list[EvidenceAssessment],
    conflicts: list[EvidenceConflict],
) -> set[str]:
    assessment_by_id = {
        item.evidence_id: item
        for item in assessments
        if item.usable
    }
    claim_by_id = {item.claim_id: item for item in claims}
    supported = {
        link.claim_id
        for link in links
        if link.relation == "supports"
        and link.evidence_id in assessment_by_id
        and (
            claim_by_id.get(link.claim_id) is None
            or claim_by_id[link.claim_id].importance != "critical"
            or assessment_by_id[link.evidence_id].source_grade in {"A", "B"}
        )
    }
    blocked = {
        claim_id
        for conflict in conflicts
        for claim_id in conflict.claim_ids
        if (
            not conflict.resolved
            or (
                conflict.preferred_claim_id is not None
                and claim_id != conflict.preferred_claim_id
            )
        )
    }
    known = {item.claim_id for item in claims}
    return (supported & known) - blocked


def validate_constrained_answer(
    answer: ConstrainedAnswer,
    *,
    supported_ids: set[str],
    links: list[ClaimEvidenceLink],
    assessments: list[EvidenceAssessment],
) -> list[str]:
    """确定性检查每个事实句是否只使用合法 Claim 和 Evidence。"""
    usable = {item.evidence_id for item in assessments if item.usable}
    linked = {
        (item.claim_id, item.evidence_id)
        for item in links
        if item.relation == "supports"
    }
    issues: list[str] = []
    for index, statement in enumerate(answer.statements):
        if statement.statement_type == "caveat":
            continue
        if not statement.claim_ids:
            issues.append(f"statement_{index}:claim_missing")
            continue
        if not set(statement.claim_ids) <= supported_ids:
            issues.append(f"statement_{index}:unsupported_claim")
        if not statement.evidence_ids:
            issues.append(f"statement_{index}:evidence_missing")
            continue
        for evidence_id in statement.evidence_ids:
            if evidence_id not in usable:
                issues.append(f"statement_{index}:evidence_unusable:{evidence_id}")
                continue
            if not any(
                (claim_id, evidence_id) in linked
                for claim_id in statement.claim_ids
            ):
                issues.append(f"statement_{index}:evidence_not_linked:{evidence_id}")
    return list(dict.fromkeys(issues))


def _fallback_constrained_answer(
    claims: list[Claim],
    links: list[ClaimEvidenceLink],
    assessments: list[EvidenceAssessment],
    conflicts: list[EvidenceConflict],
) -> ConstrainedAnswer:
    supported = supported_claim_ids(claims, links, assessments, conflicts)
    usable = {
        item.evidence_id: item
        for item in assessments
        if item.usable
    }
    evidence_by_claim: dict[str, list[str]] = {}
    for link in links:
        if link.claim_id in supported and link.evidence_id in usable:
            evidence_by_claim.setdefault(link.claim_id, []).append(link.evidence_id)
    statements = []
    for claim in claims:
        evidence_ids = list(dict.fromkeys(evidence_by_claim.get(claim.claim_id, [])))
        if claim.claim_id not in supported or not evidence_ids:
            continue
        confidence = min(
            usable[item].authority_score * usable[item].freshness_score
            for item in evidence_ids
        )
        statements.append(
            AnswerStatement(
                text=claim.text,
                claim_ids=[claim.claim_id],
                evidence_ids=evidence_ids,
                statement_type=(
                    "inference"
                    if claim.claim_type in {"inference", "opinion"}
                    else "fact"
                ),
                confidence=confidence,
            )
        )
    unresolved = [
        item.claim_id
        for item in claims
        if item.claim_id not in supported
    ]
    caveats = [
        f"存在未解决证据冲突：{item.claim_key}"
        for item in conflicts
        if not item.resolved
    ]
    if unresolved:
        caveats.append(f"{len(unresolved)} 条结论缺少可用证据，未写入确定性答案。")
    return ConstrainedAnswer(
        statements=statements,
        unresolved_claim_ids=unresolved,
        caveats=caveats,
    )


async def constrained_synthesis(
    claims: list[Claim],
    links: list[ClaimEvidenceLink],
    assessments: list[EvidenceAssessment],
    conflicts: list[EvidenceConflict],
    *,
    config: RunnableConfig | None = None,
    allow_llm: bool = True,
) -> ConstrainedAnswer:
    """最多两次模型调用；失败后使用只含已支持 Claim 的模板答案。"""
    fallback = _fallback_constrained_answer(
        claims,
        links,
        assessments,
        conflicts,
    )
    supported = supported_claim_ids(claims, links, assessments, conflicts)
    if not allow_llm or not settings.DEEPSEEK_API_KEY or not supported:
        return fallback

    allowed_claims = [
        item.model_dump(mode="json")
        for item in claims
        if item.claim_id in supported
    ]
    allowed_links = [
        item.model_dump(mode="json")
        for item in links
        if item.claim_id in supported
    ]
    base_prompt = (
        "仅使用给定 Claim 和 ClaimEvidenceLink 生成中文答案。每个事实或推断句必须"
        "填写 claim_ids 和 evidence_ids；不得增加输入之外的数字、实体或判断。"
        "冲突和证据不足只能写入 caveats。\n"
        f"claims={json.dumps(allowed_claims, ensure_ascii=False, default=str)}\n"
        f"links={json.dumps(allowed_links, ensure_ascii=False, default=str)}"
    )
    repair_issues: list[str] = []
    for attempt in range(2):
        prompt = base_prompt
        if repair_issues:
            prompt += (
                "\n上次输出未通过校验，请只修正这些问题："
                + ", ".join(repair_issues)
            )
        try:
            answer = await ainvoke_json_output(
                get_router_llm(),
                ConstrainedAnswer,
                [
                    (
                        "system",
                        "你是受约束金融答案综合器，不得越过提供的 Claim—Evidence 边界。"
                        f"\n{_CONSTRAINED_SYNTHESIS_JSON_CONTRACT}",
                    ),
                    ("human", prompt),
                ],
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "constrained synthesis fallback attempt={} error={}",
                attempt + 1,
                type(exc).__name__,
            )
            break
        repair_issues = validate_constrained_answer(
            answer,
            supported_ids=supported,
            links=links,
            assessments=assessments,
        )
        if not repair_issues:
            return answer
    return fallback


def assess_all_evidence(
    evidence: list[Evidence],
    profile: RequestProfile | None,
) -> list[EvidenceAssessment]:
    """按请求新鲜度要求批量评估证据。"""
    return [
        assess_evidence(
            item,
            freshness_required=bool(profile and profile.freshness_required),
        )
        for item in evidence
    ]


__all__ = [
    "assess_all_evidence",
    "assess_evidence",
    "build_claim_evidence_map",
    "constrained_synthesis",
    "detect_claim_conflicts",
    "source_grade",
    "supported_claim_ids",
    "validate_constrained_answer",
]
