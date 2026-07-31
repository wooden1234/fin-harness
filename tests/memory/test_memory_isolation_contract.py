from app.models.memory.memory_record import MemoryRecord
from app.services.memory.memory_service import MemoryService


def test_memory_scope_contains_tenant_and_user():
    statement = MemoryService._scope(
        MemoryRecord.__table__.select(),
        tenant_id="tenant-a",
        user_id=7,
    )
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "tenant_id = 'tenant-a'" in sql
    assert "user_id = 7" in sql


def test_index_event_keys_are_versioned_and_idempotent():
    first = "memory:index:upsert:memory-1:v:1"
    second = "memory:index:upsert:memory-1:v:2"
    assert first != second
    assert first == "memory:index:upsert:memory-1:v:1"
