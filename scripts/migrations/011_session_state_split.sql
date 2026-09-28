CREATE SCHEMA IF NOT EXISTS app;

ALTER TABLE app.agent_sessions
    ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'idle',
    ADD COLUMN IF NOT EXISTS current_run_id VARCHAR(36),
    ADD COLUMN IF NOT EXISTS current_turn INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS current_step INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_event_seq BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS pending_approval_id VARCHAR(36),
    ADD COLUMN IF NOT EXISTS latest_compaction_id VARCHAR(36),
    ADD COLUMN IF NOT EXISTS last_compacted_seq BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS latest_snapshot_seq BIGINT NOT NULL DEFAULT 0;

ALTER TABLE app.agent_session_leases
    ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS app.session_event_log (
    event_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    seq BIGINT NOT NULL,
    event_type VARCHAR(80) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    run_id VARCHAR(36),
    turn INTEGER,
    step INTEGER,
    causation_seq BIGINT,
    correlation_id VARCHAR(36),
    surface_op VARCHAR(16) NOT NULL DEFAULT 'none',
    source_event_seqs JSONB NOT NULL DEFAULT '[]'::jsonb,
    visibility VARCHAR(16) NOT NULL DEFAULT 'internal',
    payload_kind VARCHAR(32),
    payload_id VARCHAR(128),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_session_event_log_session_seq ON app.session_event_log(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_session_event_log_run ON app.session_event_log(run_id, seq);

CREATE TABLE IF NOT EXISTS app.session_messages (
    message_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    event_seq BIGINT NOT NULL,
    run_id VARCHAR(36),
    turn INTEGER,
    step INTEGER,
    role VARCHAR(20) NOT NULL,
    source VARCHAR(32),
    content TEXT NOT NULL DEFAULT '',
    tool_calls JSONB,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    published BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, event_seq)
);
CREATE INDEX IF NOT EXISTS ix_session_messages_surface ON app.session_messages(session_id, event_seq);

CREATE TABLE IF NOT EXISTS app.session_model_calls (
    model_call_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    run_id VARCHAR(36) NOT NULL,
    turn INTEGER NOT NULL,
    step INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL,
    system_prompt_hash VARCHAR(64),
    tools_hash VARCHAR(64),
    tools_ref VARCHAR(256),
    source_from_seq BIGINT,
    source_to_seq BIGINT,
    request_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_message_id VARCHAR(36),
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER,
    finish_reason VARCHAR(32),
    error_code VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_session_model_calls_run ON app.session_model_calls(session_id, run_id, step);

CREATE TABLE IF NOT EXISTS app.session_tool_calls (
    tool_call_id VARCHAR(36) PRIMARY KEY,
    provider_call_id VARCHAR(128),
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    run_id VARCHAR(36) NOT NULL,
    turn INTEGER NOT NULL,
    step INTEGER NOT NULL,
    tool_name VARCHAR(128) NOT NULL,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    arguments_hash VARCHAR(64) NOT NULL,
    idempotency_key VARCHAR(160) NOT NULL,
    status VARCHAR(32) NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    call_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_preview JSONB,
    model_content TEXT,
    result_ref VARCHAR(256),
    result_hash VARCHAR(64),
    evidence_id VARCHAR(128),
    error_code VARCHAR(128),
    error_class VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE (session_id, provider_call_id),
    UNIQUE (idempotency_key)
);
CREATE INDEX IF NOT EXISTS ix_session_tool_calls_status ON app.session_tool_calls(session_id, status);

CREATE TABLE IF NOT EXISTS app.session_tool_attempts (
    attempt_id VARCHAR(36) PRIMARY KEY,
    tool_call_id VARCHAR(36) NOT NULL REFERENCES app.session_tool_calls(tool_call_id) ON DELETE CASCADE,
    attempt_no INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL,
    error_code VARCHAR(128),
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_content TEXT,
    result_hash VARCHAR(64),
    latency_ms INTEGER,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE (tool_call_id, attempt_no)
);

ALTER TABLE app.session_tool_attempts
    ADD COLUMN IF NOT EXISTS event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS model_content TEXT,
    ADD COLUMN IF NOT EXISTS result_hash VARCHAR(64);

CREATE TABLE IF NOT EXISTS app.session_approvals (
    approval_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    run_id VARCHAR(36) NOT NULL,
    tool_call_id VARCHAR(36),
    status VARCHAR(32) NOT NULL,
    requested_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    decision VARCHAR(16),
    decided_by VARCHAR(64),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_session_approvals_status ON app.session_approvals(session_id, status);

CREATE TABLE IF NOT EXISTS app.session_compactions (
    compaction_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    run_id VARCHAR(36) NOT NULL,
    turn INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL,
    source_from_seq BIGINT NOT NULL DEFAULT 0,
    source_to_seq BIGINT NOT NULL DEFAULT 0,
    source_event_seqs JSONB,
    schema_version INTEGER NOT NULL DEFAULT 1,
    summary TEXT NOT NULL DEFAULT '',
    structured_summary JSONB,
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary_tokens INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    committed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_session_compactions_latest ON app.session_compactions(session_id, status, source_to_seq);

CREATE TABLE IF NOT EXISTS app.session_todo_snapshots (
    snapshot_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    event_seq BIGINT NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    turn INTEGER NOT NULL,
    version INTEGER NOT NULL,
    todos JSONB NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, turn, version)
);

CREATE TABLE IF NOT EXISTS app.session_state_snapshots (
    snapshot_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    snapshot_seq BIGINT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    current_run_id VARCHAR(36),
    current_turn INTEGER,
    current_step INTEGER,
    run_status VARCHAR(32),
    pending_approval_id VARCHAR(36),
    pending_call_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    published_message_id VARCHAR(36),
    latest_compaction_id VARCHAR(36),
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, snapshot_seq)
);
