-- Agent/工具流上下文压缩及研究过程事件；不保存原始消息和隐藏推理。

CREATE TABLE IF NOT EXISTS app.agent_run_events (
    id BIGSERIAL PRIMARY KEY,
    event_key VARCHAR(160) NOT NULL UNIQUE,
    tenant_id VARCHAR(36) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES app.users(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES app.conversations(id) ON DELETE CASCADE,
    run_id VARCHAR(36) NOT NULL REFERENCES app.agent_runs(id) ON DELETE CASCADE,
    task_id VARCHAR(128),
    agent_id VARCHAR(128),
    space_type VARCHAR(32) NOT NULL,
    space_id VARCHAR(255) NOT NULL,
    parent_space_id VARCHAR(255),
    event_type VARCHAR(80) NOT NULL,
    estimated_tokens INTEGER,
    actual_input_tokens INTEGER,
    effective_limit INTEGER,
    tokens_before INTEGER,
    tokens_after INTEGER,
    compaction_round_count INTEGER NOT NULL DEFAULT 0,
    summary_attempt_count INTEGER NOT NULL DEFAULT 0,
    snip_count INTEGER NOT NULL DEFAULT 0,
    provider_retry_count INTEGER NOT NULL DEFAULT 0,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_run_events_run_id
    ON app.agent_run_events (run_id, id);

CREATE INDEX IF NOT EXISTS ix_agent_run_events_space
    ON app.agent_run_events (space_type, space_id, id);
