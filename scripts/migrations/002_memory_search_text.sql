-- 为长期记忆增加用于语义检索的规范化文本。
ALTER TABLE app.memory_records
    ADD COLUMN IF NOT EXISTS search_text TEXT;

UPDATE app.memory_records
SET search_text = COALESCE(display_text, memory_key)
WHERE search_text IS NULL;

ALTER TABLE app.memory_records
    ALTER COLUMN search_text SET NOT NULL;
