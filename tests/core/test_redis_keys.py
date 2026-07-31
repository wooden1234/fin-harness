import pytest

from app.core.redis_keys import RedisKeyBuilder


def test_redis_key_builder_adds_environment_and_user_scope():
    builder = RedisKeyBuilder(prefix="fin", environment="test")

    key = builder.user("rate", "tenant-a", 7, "agent-query")

    assert key.value == "fin:test:rate:tenant:tenant-a:user:7:agent-query"


def test_redis_key_builder_rejects_manual_delimiters_and_empty_segments():
    builder = RedisKeyBuilder(prefix="fin", environment="test")

    with pytest.raises(ValueError):
        builder.build("cache:unsafe", "value")
    with pytest.raises(ValueError):
        builder.build("cache", "")


def test_redis_key_builder_hashes_long_or_sensitive_input():
    digest = RedisKeyBuilder.digest("用户原始问题")

    assert len(digest) == 64
    assert "用户" not in digest
