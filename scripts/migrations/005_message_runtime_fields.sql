-- 为消息补齐运行链路与幂等字段，兼容已有历史记录
ALTER TABLE app.messages
    ADD COLUMN IF NOT EXISTS run_id VARCHAR(36);

ALTER TABLE app.messages
    ADD COLUMN IF NOT EXISTS sequence_no INTEGER;

ALTER TABLE app.messages
    ADD COLUMN IF NOT EXISTS client_message_id VARCHAR(128);

CREATE INDEX IF NOT EXISTS ix_messages_run_id
    ON app.messages (run_id);

CREATE INDEX IF NOT EXISTS ix_messages_client_message_id
    ON app.messages (client_message_id);

CREATE INDEX IF NOT EXISTS ix_messages_conversation_sequence
    ON app.messages (conversation_id, sequence_no);
