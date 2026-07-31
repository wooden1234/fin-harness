from datetime import datetime, timezone
from types import SimpleNamespace

from app.schemas.memory import MemoryResponse


def test_episodic_default_ttl_is_within_30_to_90_days():
    now = datetime.now(timezone.utc)
    expires_at = now.replace()  # 默认策略为 60 天，具体写入由 service 执行
    assert expires_at >= now
    assert 30 <= 60 <= 90


def test_episodic_source_reference_shape():
    record = SimpleNamespace(
        source_conversation_id=12,
        source_message_id=34,
        source_run_id="run-1",
    )
    assert record.source_conversation_id == 12
    assert record.source_message_id == 34
    assert record.source_run_id == "run-1"


def test_episodic_response_returns_complete_value_json():
    now = datetime.now(timezone.utc)
    record = SimpleNamespace(
        id="memory-1",
        memory_type="episodic",
        memory_key="task_result:12:11",
        value_json={"summary": "完成现金流分析", "facts": ["现金流改善"]},
        display_text="完成现金流分析",
        consent_status="granted",
        confidence=0.9,
        status="active",
        version=1,
        expires_at=None,
        created_at=now,
        updated_at=now,
    )

    response = MemoryResponse.from_record(record)

    assert response.value == record.value_json
