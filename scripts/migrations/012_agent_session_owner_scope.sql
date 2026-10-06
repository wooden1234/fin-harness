CREATE SCHEMA IF NOT EXISTS app;

-- A conversation identifier is only meaningful in its tenant and owner scope.
ALTER TABLE app.agent_sessions
    DROP CONSTRAINT IF EXISTS uq_agent_sessions_conversation_id;

ALTER TABLE app.agent_sessions
    ADD CONSTRAINT uq_agent_sessions_owner_conversation
    UNIQUE (tenant_id, user_id, conversation_id);
