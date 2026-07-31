import pytest

from app.api.memories import router as memory_router
from app.services.memory.memory_extraction import (
    PreferenceExtractionOutput,
    extract_preference,
)
from app.services.memory.memory_service import MemoryService


class FakeStructuredRunner:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, messages):
        _ = messages
        return self.payload


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.structured_calls = 0

    def with_structured_output(self, schema, *, method):
        assert schema is PreferenceExtractionOutput
        assert method == "json_mode"
        self.structured_calls += 1
        return FakeStructuredRunner(self.payload)


@pytest.mark.asyncio
async def test_explicit_rule_short_circuits_llm():
    llm = FakeLLM({})

    result = await extract_preference("请记住以后用中文回答", llm=llm)

    assert result is not None
    assert result.memory_key == "response_language"
    assert result.value == "zh-CN"
    assert result.source == "explicit_rule"
    assert llm.structured_calls == 0


@pytest.mark.asyncio
async def test_preference_rule_short_circuits_llm():
    llm = FakeLLM({})

    result = await extract_preference("我喜欢表格展示", llm=llm)

    assert result is not None
    assert result.memory_key == "preferred_output_format"
    assert result.value == "table"
    assert result.source == "preference_rule"
    assert llm.structured_calls == 0


@pytest.mark.asyncio
async def test_llm_runs_only_after_rules_miss_and_passes_policy_validation():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "response_detail_level",
            "value": "brief",
            "confidence": 0.92,
            "evidence": "以后回答直接给结论",
        }
    )

    result = await extract_preference("以后回答直接给结论", llm=llm)

    assert result is not None
    assert result.memory_key == "response_detail_level"
    assert result.value == "brief"
    assert result.source == "llm"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_llm_output_is_rejected_by_final_policy_validation():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "holdings",
            "value": "600000",
            "confidence": 0.99,
            "evidence": "长期持有 600000",
        }
    )

    result = await extract_preference("我长期持有 600000", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_llm_temporary_request_is_not_persisted():
    llm = FakeLLM(
        {
            "has_preference": False,
            "memory_key": "",
            "value": "",
            "confidence": 0.0,
            "evidence": "",
        }
    )

    result = await extract_preference("这次请用英文回答", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_high_confidence_temporary_preference_is_rejected():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "response_language",
            "value": "en-US",
            "confidence": 0.99,
            "evidence": "这次请用英文回答",
        }
    )

    result = await extract_preference("这次请用英文回答", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_fabricated_evidence_is_rejected():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "response_detail_level",
            "value": "brief",
            "confidence": 0.99,
            "evidence": "以后请简短回答",
        }
    )

    result = await extract_preference("以后回答直接给结论", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_preference_without_persistent_marker_is_rejected():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "response_detail_level",
            "value": "brief",
            "confidence": 0.99,
            "evidence": "回答直接给结论",
        }
    )

    result = await extract_preference("回答直接给结论", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_sensitive_source_text_is_rejected():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "response_language",
            "value": "zh-CN",
            "confidence": 0.99,
            "evidence": "以后用中文回答",
        }
    )

    result = await extract_preference(
        "我的账户余额是十万元，以后用中文回答",
        llm=llm,
    )

    assert result is None


@pytest.mark.asyncio
async def test_key_value_must_be_supported_by_evidence():
    llm = FakeLLM(
        {
            "has_preference": True,
            "memory_key": "preferred_output_format",
            "value": "table",
            "confidence": 0.99,
            "evidence": "以后输出采用 Markdown 格式",
        }
    )

    result = await extract_preference("以后输出采用 Markdown 格式", llm=llm)

    assert result is None


@pytest.mark.asyncio
async def test_temporary_marker_overrides_explicit_memory_prefix():
    llm = FakeLLM({})

    result = await extract_preference("请记住这次用中文回答", llm=llm)

    assert result is None
    assert llm.structured_calls == 0


def test_confirmation_service_and_routes_are_removed():
    assert not hasattr(MemoryService, "create_candidate")
    assert not hasattr(MemoryService, "list_candidates")
    assert not hasattr(MemoryService, "decide_candidate")

    paths = {route.path for route in memory_router.routes}
    assert "/memories/candidates" not in paths
    assert "/memories/{memory_id}/confirm" not in paths
    assert "/memories/{memory_id}/reject" not in paths
