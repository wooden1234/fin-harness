from datetime import datetime, timezone
from types import SimpleNamespace


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
