-- 阶段 6：记忆 active 唯一性、审计上下文和缓存补偿事件。
-- 生产环境请在事务中执行，并在迁移表中登记版本。

CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory
    ON app.memory_records (tenant_id, user_id, memory_type, memory_key)
    WHERE status = 'active';

ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(36);
ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS memory_type VARCHAR(32);
ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS memory_key VARCHAR(64);
ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS agent_id VARCHAR(128);
ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS task_id VARCHAR(128);
ALTER TABLE app.audit_logs ADD COLUMN IF NOT EXISTS trace_id VARCHAR(128);

CREATE INDEX IF NOT EXISTS ix_audit_logs_memory_scope
    ON app.audit_logs (tenant_id, user_id, resource_type, created_at);
