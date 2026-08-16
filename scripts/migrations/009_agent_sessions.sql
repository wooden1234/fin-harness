CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.agent_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(64) UNIQUE,
    tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
    user_id VARCHAR(36) NOT NULL,
    next_seq INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_agent_sessions_user
    ON app.agent_sessions (tenant_id, user_id);

CREATE TABLE IF NOT EXISTS app.session_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    event_id VARCHAR(36) NOT NULL UNIQUE,
    event_type VARCHAR(80) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    run_id VARCHAR(36),
    turn INTEGER,
    step INTEGER,
    causation_seq INTEGER,
    correlation_id VARCHAR(36),
    surface_op VARCHAR(16) NOT NULL DEFAULT 'none',
    source_event_seqs JSONB NOT NULL DEFAULT '[]'::jsonb,
    visibility VARCHAR(16) NOT NULL DEFAULT 'internal',
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, seq)
);

CREATE INDEX IF NOT EXISTS ix_session_events_session_seq
    ON app.session_events (session_id, seq);

CREATE TABLE IF NOT EXISTS app.agent_session_leases (
    session_id VARCHAR(36) PRIMARY KEY REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE,
    owner_id VARCHAR(64) NOT NULL,
    token VARCHAR(36) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
