import pytest

from app.services.memory import memory_episodic_extraction as extraction


class FakeStructuredRunner:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, messages):
        assert messages
        return self.payload


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, schema, *, method):
        assert schema is extraction.EpisodicExtractionOutput
        assert method == "json_mode"
        return FakeStructuredRunner(self.payload)


def test_obvious_state_change_is_sync_trigger():
    assert extraction.detect_important_state_change(
        "不再使用纯向量检索，改成混合检索。"
    )
    assert not extraction.detect_important_state_change("把回答语言改成英文")


def test_capacity_threshold_forces_post_turn_trigger(monkeypatch):
    monkeypatch.setattr(
        extraction.settings,
        "EPISODIC_MEMORY_TURN_THRESHOLD",
        10,
    )
    decision = extraction.decide_post_turn_trigger(
        query="继续分析",
        final_response="这里是有实质内容的分析结果。" * 20,
        execution_status="completed",
        task_count=0,
        turn_count=10,
        uncompressed_tokens=100,
    )

    assert decision.should_enqueue is True
    assert decision.forced is True
    assert "turn_threshold" in decision.reasons


def test_completed_task_triggers_async_candidate():
    decision = extraction.decide_post_turn_trigger(
        query="分析贵州茅台现金流",
        final_response="已经完成现金流、利润质量和风险因素分析。" * 20,
        execution_status="completed",
        task_count=1,
        turn_count=1,
        uncompressed_tokens=300,
    )

    assert decision.should_enqueue is True
    assert decision.forced is False
    assert decision.reasons == ("task_completed",)


@pytest.mark.asyncio
async def test_llm_summary_requires_source_evidence(monkeypatch):
    monkeypatch.setattr(
        extraction.settings,
        "EPISODIC_MEMORY_MIN_CONFIDENCE",
        0.80,
    )
    source = "用户: 不再使用纯向量检索，改成混合检索。"
    result = await extraction.extract_episodic_memory(
        source,
        trigger_reason="explicit_state_change",
        forced=True,
        expected_event_type="state_change",
        llm=FakeLLM(
            {
                "should_store": True,
                "event_type": "state_change",
                "subject_key": "retrieval_strategy",
                "topic": "检索策略变化",
                "summary": "用户将检索策略从纯向量检索调整为混合检索。",
                "facts": ["旧策略是纯向量检索", "新策略是混合检索"],
                "conclusion": "后续任务使用混合检索。",
                "evidence": ["不再使用纯向量检索，改成混合检索"],
                "confidence": 0.95,
            }
        ),
    )

    assert result is not None
    assert result.event_type == "state_change"
    assert result.subject_key == "retrieval_strategy"
    assert result.quality_score >= 0.80


@pytest.mark.asyncio
async def test_fabricated_episodic_evidence_is_rejected():
    result = await extraction.extract_episodic_memory(
        "用户: 分析贵州茅台现金流。",
        trigger_reason="task_completed",
        llm=FakeLLM(
            {
                "should_store": True,
                "event_type": "task_result",
                "subject_key": "stock_analysis",
                "topic": "贵州茅台分析",
                "summary": "用户完成了贵州茅台现金流和毛利率分析。",
                "facts": ["分析了现金流"],
                "conclusion": "后续继续跟踪。",
                "evidence": ["用户已经完成毛利率分析"],
                "confidence": 0.99,
            }
        ),
    )

    assert result is None
