"""上下文监控事件的脱敏和聚合契约测试。"""

from types import SimpleNamespace

from app.services.agent.context_event_service import ContextEventService, _safe_details


def test_event_details_drop_raw_content_and_hidden_reasoning() -> None:
    details = _safe_details(
        {
            "content_hash": "sha256:abc",
            "evidence_ids": ["ev-1"],
            "raw_message": "不应落库的正文",
            "tool_result": {"secret": "不应落库"},
            "hidden_reasoning": "不应落库",
        }
    )

    assert details == {
        "content_hash": "sha256:abc",
        "evidence_ids": ["ev-1"],
    }


async def test_metrics_aggregate_named_context_counters(monkeypatch) -> None:
    events = [
        SimpleNamespace(
            space_id="research_run:run-1",
            space_type="research_run",
            estimated_tokens=100,
            actual_input_tokens=120,
            compaction_round_count=1,
            summary_attempt_count=2,
            snip_count=1,
            provider_retry_count=0,
            event_type="context.snip_applied",
            details={},
        ),
        SimpleNamespace(
            space_id="research_run:run-1",
            space_type="research_run",
            estimated_tokens=110,
            actual_input_tokens=140,
            compaction_round_count=1,
            summary_attempt_count=2,
            snip_count=1,
            provider_retry_count=1,
            event_type="context.provider_overflow",
            details={},
        ),
        SimpleNamespace(
            space_id="research_run:run-1",
            space_type="research_run",
            estimated_tokens=90,
            actual_input_tokens=None,
            compaction_round_count=2,
            summary_attempt_count=2,
            snip_count=1,
            provider_retry_count=1,
            event_type="context.admission_rejected",
            details={},
        ),
    ]

    async def _list_events(cls, **kwargs):
        del cls, kwargs
        return events

    monkeypatch.setattr(ContextEventService, "list_events", classmethod(_list_events))

    result = await ContextEventService.metrics(
        run_id="run-1",
        tenant_id="tenant-1",
        user_id=1,
    )

    assert result["context_provider_overflow_total"] == 1
    assert result["context_post_compaction_overflow_total"] == 1
    assert result["context_snip_total"] == 1
    assert result["context_admission_rejected_total"] == 1
    assert result["context_estimate_underflow_total"] == 2
    assert result["spaces"]["research_run:run-1"]["compaction_round_count"] == 2
