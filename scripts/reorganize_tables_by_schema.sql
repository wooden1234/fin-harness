-- 按业务域整理 public 中的表到独立 schema。
--
-- 目标分组：
--   fin_core:
--     financial_companies
--     financial_metrics
--     annual_report_documents
--     annual_financial_tables
--     annual_financial_facts
--
--   fin_legacy:
--     annual_financial_facts_legacy
--
--   rag:
--     public 下所有 *_vectors
--
--   app:
--     users
--     conversations
--     messages
--
--   runtime:
--     public 下所有 checkpoint_*
--
-- 注意：
-- 1. 本脚本只负责整理数据库对象，不修改应用代码。
-- 2. 当前仓库中的 ORM / PGVectorStore 仍默认使用 public；
--    真正执行迁移前，需要同步评估 search_path、ORM schema、向量库 schema_name。
-- 3. 本脚本是幂等的：已移动过的表不会重复报错。

BEGIN;

CREATE SCHEMA IF NOT EXISTS fin_core;
CREATE SCHEMA IF NOT EXISTS fin_legacy;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS app;
CREATE SCHEMA IF NOT EXISTS runtime;

CREATE OR REPLACE FUNCTION public._move_table_if_exists(
    src_schema text,
    dst_schema text,
    table_name text
) RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_tables
        WHERE schemaname = src_schema
          AND tablename = table_name
    ) THEN
        EXECUTE format(
            'ALTER TABLE %I.%I SET SCHEMA %I',
            src_schema,
            table_name,
            dst_schema
        );
        RAISE NOTICE 'moved %.% -> %', src_schema, table_name, dst_schema;
    ELSE
        RAISE NOTICE 'skip %.% (not found)', src_schema, table_name;
    END IF;
END;
$$;

-- fin_core
SELECT public._move_table_if_exists('public', 'fin_core', 'financial_companies');
SELECT public._move_table_if_exists('public', 'fin_core', 'financial_metrics');
SELECT public._move_table_if_exists('public', 'fin_core', 'annual_report_documents');
SELECT public._move_table_if_exists('public', 'fin_core', 'annual_financial_tables');
SELECT public._move_table_if_exists('public', 'fin_core', 'annual_financial_facts');

-- fin_legacy
SELECT public._move_table_if_exists('public', 'fin_legacy', 'annual_financial_facts_legacy');

-- app
SELECT public._move_table_if_exists('public', 'app', 'users');
SELECT public._move_table_if_exists('public', 'app', 'conversations');
SELECT public._move_table_if_exists('public', 'app', 'messages');

DO $$
DECLARE
    item record;
BEGIN
    -- rag：所有 *_vectors
    FOR item IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename LIKE '%\_vectors' ESCAPE '\'
    LOOP
        EXECUTE format(
            'ALTER TABLE %I.%I SET SCHEMA %I',
            'public',
            item.tablename,
            'rag'
        );
        RAISE NOTICE 'moved public.% -> rag', item.tablename;
    END LOOP;

    -- runtime：所有 checkpoint_*
    FOR item IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename LIKE 'checkpoint\_%' ESCAPE '\'
    LOOP
        EXECUTE format(
            'ALTER TABLE %I.%I SET SCHEMA %I',
            'public',
            item.tablename,
            'runtime'
        );
        RAISE NOTICE 'moved public.% -> runtime', item.tablename;
    END LOOP;
END;
$$;

DROP FUNCTION public._move_table_if_exists(text, text, text);

COMMIT;
