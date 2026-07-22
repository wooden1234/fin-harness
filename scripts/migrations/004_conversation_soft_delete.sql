-- 补齐会话软删除字段，兼容早期 conversations 表结构。
ALTER TABLE app.conversations
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

ALTER TABLE app.conversations
    ADD COLUMN IF NOT EXISTS deleted_by INTEGER;

CREATE INDEX IF NOT EXISTS ix_conversations_deleted_at
    ON app.conversations (deleted_at);
