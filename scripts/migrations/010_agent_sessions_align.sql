-- Align leftover agent_sessions schema (PK `id`) to harness-agent-loop (`session_id`).
-- 009 used CREATE TABLE IF NOT EXISTS, so an older table was left unchanged.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'agent_sessions' AND column_name = 'id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'agent_sessions' AND column_name = 'session_id'
    ) THEN
        ALTER TABLE app.session_events DROP CONSTRAINT IF EXISTS session_events_session_id_fkey;
        ALTER TABLE app.agent_session_leases DROP CONSTRAINT IF EXISTS agent_session_leases_session_id_fkey;
        ALTER TABLE app.agent_approval_projection DROP CONSTRAINT IF EXISTS agent_approval_projection_session_id_fkey;
        ALTER TABLE app.agent_sessions DROP CONSTRAINT IF EXISTS agent_sessions_conversation_id_fkey;
        ALTER TABLE app.agent_sessions DROP CONSTRAINT IF EXISTS agent_sessions_user_id_fkey;
        DROP INDEX IF EXISTS app.uq_agent_sessions_conversation;
        DROP INDEX IF EXISTS app.ix_agent_sessions_tenant;
        DROP INDEX IF EXISTS app.ix_agent_sessions_user;

        ALTER TABLE app.agent_sessions RENAME COLUMN id TO session_id;
        ALTER TABLE app.agent_sessions
            ALTER COLUMN conversation_id TYPE VARCHAR(64) USING conversation_id::text,
            ALTER COLUMN user_id TYPE VARCHAR(36) USING user_id::text;

        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_sessions_conversation_id
            ON app.agent_sessions (conversation_id);
        CREATE INDEX IF NOT EXISTS ix_agent_sessions_user
            ON app.agent_sessions (tenant_id, user_id);

        ALTER TABLE app.session_events
            ADD CONSTRAINT session_events_session_id_fkey
            FOREIGN KEY (session_id) REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE;
        ALTER TABLE app.agent_session_leases
            ADD CONSTRAINT agent_session_leases_session_id_fkey
            FOREIGN KEY (session_id) REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE;
        ALTER TABLE app.agent_approval_projection
            ADD CONSTRAINT agent_approval_projection_session_id_fkey
            FOREIGN KEY (session_id) REFERENCES app.agent_sessions(session_id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'app' AND table_name = 'session_events'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'session_events' AND column_name = 'id'
    ) THEN
        ALTER TABLE app.session_events ADD COLUMN id INTEGER;
        UPDATE app.session_events SET id = seq
        WHERE id IS NULL;
        CREATE SEQUENCE IF NOT EXISTS app.session_events_id_seq;
        PERFORM setval(
            'app.session_events_id_seq',
            GREATEST(COALESCE((SELECT MAX(id) FROM app.session_events), 1), 1),
            true
        );
        ALTER TABLE app.session_events
            ALTER COLUMN id SET DEFAULT nextval('app.session_events_id_seq');
        ALTER SEQUENCE app.session_events_id_seq OWNED BY app.session_events.id;
        ALTER TABLE app.session_events ALTER COLUMN id SET NOT NULL;
        ALTER TABLE app.session_events DROP CONSTRAINT IF EXISTS session_events_pkey;
        ALTER TABLE app.session_events ADD PRIMARY KEY (id);
        ALTER TABLE app.session_events DROP CONSTRAINT IF EXISTS uq_session_events_session_seq;
        ALTER TABLE app.session_events
            ADD CONSTRAINT uq_session_events_session_seq UNIQUE (session_id, seq);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'agent_session_leases'
          AND column_name = 'lease_token'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'agent_session_leases'
          AND column_name = 'token'
    ) THEN
        ALTER TABLE app.agent_session_leases RENAME COLUMN lease_token TO token;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'app' AND table_name = 'agent_session_leases'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'app' AND table_name = 'agent_session_leases'
          AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE app.agent_session_leases
            ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
    END IF;
END $$;
