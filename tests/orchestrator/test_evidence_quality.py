"""Claim—Evidence 映射、证据质量和受约束综合测试。"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

from agents.final_answer.node import _filter_current_turn_citations
from agents.orchestrator.contracts import (
    AgentResult,
    AnswerStatement,
    Claim,
    ClaimEvidenceLink,
    ConstrainedAnswer,
    Evidence,
    EvidenceAssessment,
    ExecutionDecision,
    RequestProfile,
    TaskPlan,
    TaskSpec,
)
from agents.orchestrator.evidence_quality import (
    assess_evidence,
    build_claim_evidence_map,
    constrained_synthesis,
    detect_claim_conflicts,
    source_grade,
    validate_constrained_answer,
)
from agents.orchestrator.graph import map_claim_evidence, quality_gate
from agents.runtime_context import AgentRuntimeContext


def _usable(evidence_id: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_id=evidence_id,
        source_grade="A",
        authority_score=1,
        freshness_score=1,
        completeness_score=1,
        usable=True,
    )


def test_source_rating_prefers_official_disclosure() -> None:
    official = Evidence(
        evidence_id="official",
        source_type="iwencai.announcement.search",
        provider="iwencai",
        title="上市公司公告",
        content="公告正文",
    )
    unknown = Evidence(
        evidence_id="unknown",
        source_type="unknown",
        content="无法确认来源",
    )

    assert source_grade(official) == "A"
    assert source_grade(unknown) == "E"


def test_freshness_required_rejects_stale_market_evidence() -> None:
    evidence = Evidence(
        evidence_id="market-old",
        source_type="iwencai.market.query",
        provider="iwencai",
        content="旧行情",
        observed_at="2025-01-01T00:00:00+00:00",
    )

    assessment = assess_evidence(
        evidence,
        freshness_required=True,
        now=datetime(2026, 7, 28, tzinfo=timezone.utc),
    )

    assert assessment.stale is True
    assert assessment.usable is False
    assert any(
        reason.startswith("evidence_stale:")
        for reason in assessment.rejection_reasons
    )


def test_evidence_without_content_is_not_usable() -> None:
    evidence = Evidence(
        evidence_id="empty",
        source_type="financial_db",
        provider="financial_db",
    )

    assessment = assess_evidence(
        evidence,
        freshness_required=False,
    )

    assert assessment.usable is False
    assert "evidence_content_missing" in assessment.rejection_reasons


async def test_structured_claims_do_not_require_llm() -> None:
    result = AgentResult(
        task_id="financial",
        agent_id="finance_agent",
        status="completed",
        answer="营收为150亿元。",
        structured_data={
            "claims": [
                {
                    "claim_id": "revenue",
                    "text": "2025年营收为150亿元。",
                    "subject": "示例公司",
                    "predicate": "revenue",
                    "value": 150,
                    "unit": "亿元",
                    "period": "2025",
                    "evidence_ids": ["e-1"],
                }
            ]
        },
        evidence=[
            Evidence(
                evidence_id="e-1",
                task_id="financial",
                source_type="financial_db",
                provider="financial_db",
                content="2025年营收150亿元",
            )
        ],
    )

    claims, links = await build_claim_evidence_map(
        [result],
        allow_llm=False,
    )

    assert [item.claim_id for item in claims] == ["revenue"]
    assert links == [
        ClaimEvidenceLink(claim_id="revenue", evidence_id="e-1")
    ]


async def test_quality_gate_reports_stale_unsupported_claim() -> None:
    evidence = Evidence(
        evidence_id="e-old",
        task_id="finance",
        source_type="financial_db",
        provider="financial_db",
        content="旧财务数据",
        published_at="2020-01-01",
    )
    result = AgentResult(
        task_id="finance",
        agent_id="finance_agent",
        status="completed",
        answer="营收增长。",
        structured_data={
            "claims": [
                {
                    "claim_id": "revenue-growth",
                    "text": "营收增长。",
                    "evidence_ids": ["e-old"],
                }
            ]
        },
        evidence=[evidence],
    )
    state = {
            "request_profile": RequestProfile(
                original_query="最新营收",
                freshness_required=True,
                execution=ExecutionDecision(
                    mode="structured_finance",
                    budget_tier="standard",
                    allowed_capabilities=["financial_query"],
                    data_sources=["finance_rag"],
                ),
            ),
        "task_plan": TaskPlan(
            plan_id="p1",
            query="最新营收",
            tasks=[
                TaskSpec(
                    task_id="finance",
                    objective="最新营收",
                    agent_id="finance_agent",
                )
            ],
        ),
        "agent_results": [result],
        "evidence": [evidence],
    }

    mapping = await map_claim_evidence(state)
    report_update = await quality_gate({**state, **mapping})
    report = report_update["quality_report"]

    assert report.passed is False
    assert report.unsupported_claim_ids == ["revenue-growth"]
    assert report.stale_evidence_ids == ["e-old"]
    assert report.claim_coverage == 0


async def test_claim_extraction_skips_llm_after_soft_deadline(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    async def fake_build(results, *, config=None, allow_llm=True):
        del results, config
        calls.append(allow_llm)
        return [], []

    monkeypatch.setattr(
        "agents.orchestrator.graph.build_claim_evidence_map",
        fake_build,
    )
    context = AgentRuntimeContext(
        started_monotonic=time.monotonic() - 2,
    )
    context.configure_budget(
        budget_tier="light",
        soft_seconds=1,
        hard_seconds=10,
        unit_timeouts={},
    )

    await map_claim_evidence(
        {"agent_results": []},
        runtime=SimpleNamespace(context=context),
    )

    assert calls == [False]


def test_conflict_detection_normalizes_financial_units() -> None:
    equivalent = [
        Claim(
            claim_id="c1",
            text="营收1.5亿元",
            subject="示例公司",
            predicate="revenue",
            value=1.5,
            unit="亿元",
            period="2025",
        ),
        Claim(
            claim_id="c2",
            text="营收150000000元",
            subject="示例公司",
            predicate="revenue",
            value=150000000,
            unit="元",
            period="2025",
        ),
    ]
    conflicting = equivalent + [
        Claim(
            claim_id="c3",
            text="营收1.4亿元",
            subject="示例公司",
            predicate="revenue",
            value=1.4,
            unit="亿元",
            period="2025",
        )
    ]

    assert detect_claim_conflicts(equivalent, []) == []
    conflicts = detect_claim_conflicts(conflicting, [])
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "hard_conflict"
    assert conflicts[0].resolved is False


def test_temporal_conflict_prefers_latest_claim() -> None:
    claims = [
        Claim(
            claim_id="old",
            text="旧值",
            subject="示例公司",
            predicate="rating",
            value="增持",
            as_of="2026-01-01",
        ),
        Claim(
            claim_id="new",
            text="新值",
            subject="示例公司",
            predicate="rating",
            value="中性",
            as_of="2026-07-01",
        ),
    ]

    conflict = detect_claim_conflicts(claims, [])[0]

    assert conflict.conflict_type == "temporal_update"
    assert conflict.resolved is True
    assert conflict.preferred_claim_id == "new"


async def test_constrained_synthesis_only_uses_supported_claims() -> None:
    claims = [
        Claim(claim_id="supported", text="已核验事实"),
        Claim(claim_id="unsupported", text="无证据事实"),
    ]
    links = [
        ClaimEvidenceLink(
            claim_id="supported",
            evidence_id="e-1",
        )
    ]

    answer = await constrained_synthesis(
        claims,
        links,
        [_usable("e-1")],
        [],
        allow_llm=False,
    )

    assert [item.text for item in answer.statements] == ["已核验事实"]
    assert answer.unresolved_claim_ids == ["unsupported"]


def test_constrained_answer_rejects_unlinked_citation() -> None:
    answer = ConstrainedAnswer(
        statements=[
            AnswerStatement(
                text="事实",
                claim_ids=["c-1"],
                evidence_ids=["e-other"],
            )
        ]
    )

    issues = validate_constrained_answer(
        answer,
        supported_ids={"c-1"},
        links=[
            ClaimEvidenceLink(
                claim_id="c-1",
                evidence_id="e-1",
            )
        ],
        assessments=[_usable("e-1"), _usable("e-other")],
    )

    assert "statement_0:evidence_not_linked:e-other" in issues


def test_final_answer_only_keeps_evidence_used_by_synthesis() -> None:
    state = {
        "constrained_answer": ConstrainedAnswer(
            statements=[
                AnswerStatement(
                    text="事实",
                    claim_ids=["c-1"],
                    evidence_ids=["e-used"],
                )
            ]
        ),
        "evidence": [
            Evidence(
                evidence_id="e-used",
                task_id="research",
                source_type="financial_db",
            ),
            Evidence(
                evidence_id="e-unused",
                task_id="research",
                source_type="financial_db",
            ),
        ],
    }
    citations = [
        {
            "evidence_id": "e-used",
            "source": "数据库",
            "source_type": "financial_db",
            "sub_task_id": "research",
        },
        {
            "evidence_id": "e-unused",
            "source": "数据库",
            "source_type": "financial_db",
            "sub_task_id": "research",
        },
    ]

    assert _filter_current_turn_citations(state, citations) == [citations[0]]
