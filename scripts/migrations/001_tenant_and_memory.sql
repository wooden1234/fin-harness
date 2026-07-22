-- 租户上下文与长期记忆基础表；生产环境请在事务中执行并记录版本。
CREATE SCHEMA IF NOT EXISTS app;

ALTER TABLE app.users ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(36);
UPDATE app.users SET tenant_id = 'default' WHERE tenant_id IS NULL;
ALTER TABLE app.users ALTER COLUMN tenant_id SET DEFAULT 'default';
ALTER TABLE app.users ALTER COLUMN tenant_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS ix_users_tenant_id ON app.users (tenant_id);

ALTER TABLE app.conversations ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(36);
UPDATE app.conversations SET tenant_id = 'default' WHERE tenant_id IS NULL;
ALTER TABLE app.conversations ALTER COLUMN tenant_id SET DEFAULT 'default';
ALTER TABLE app.conversations ALTER COLUMN tenant_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS ix_conversations_tenant_id ON app.conversations (tenant_id);

ALTER TABLE app.agent_runs ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(36);
UPDATE app.agent_runs SET tenant_id = 'default' WHERE tenant_id IS NULL;
ALTER TABLE app.agent_runs ALTER COLUMN tenant_id SET DEFAULT 'default';
ALTER TABLE app.agent_runs ALTER COLUMN tenant_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS ix_agent_runs_tenant_id ON app.agent_runs (tenant_id);
ALTER TABLE app.checkpoint_registry ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(36);
UPDATE app.checkpoint_registry SET tenant_id = 'default' WHERE tenant_id IS NULL;
ALTER TABLE app.checkpoint_registry ALTER COLUMN tenant_id SET DEFAULT 'default';
ALTER TABLE app.checkpoint_registry ALTER COLUMN tenant_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS ix_checkpoint_registry_tenant_id
    ON app.checkpoint_registry (tenant_id);

CREATE TABLE IF NOT EXISTS app.memory_records (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
    memory_type VARCHAR(32) NOT NULL,
    memory_key VARCHAR(64) NOT NULL,
    value_json JSON NOT NULL,
    display_text TEXT NOT NULL,
    provenance_json JSON NOT NULL,
    confidence NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    consent_status VARCHAR(16) NOT NULL DEFAULT 'granted',
    consented_at TIMESTAMPTZ,
    withdrawn_at TIMESTAMPTZ,
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id VARCHAR(36),
    source_conversation_id INTEGER,
    source_message_id INTEGER,
    source_run_id VARCHAR(36),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    last_recalled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_memory_records_scope_status
    ON app.memory_records (tenant_id, user_id, memory_type, status);
CREATE INDEX IF NOT EXISTS ix_memory_records_expiry
    ON app.memory_records (status, expires_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory
    ON app.memory_records (tenant_id, user_id, memory_type, memory_key)
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS app.memory_events (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(36) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
    memory_id VARCHAR(36) REFERENCES app.memory_records(id) ON DELETE SET NULL,
    event_type VARCHAR(32) NOT NULL,
    event_key VARCHAR(128) NOT NULL UNIQUE,
    payload_json JSON NOT NULL,
    actor_type VARCHAR(16) NOT NULL DEFAULT 'user',
    actor_id VARCHAR(64),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_memory_events_scope
    ON app.memory_events (tenant_id, user_id, created_at);
