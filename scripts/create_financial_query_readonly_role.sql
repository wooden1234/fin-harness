-- 使用管理员或 fin 账号执行：
-- PGPASSWORD="$POSTGRES_ADMIN_PASSWORD" psql "postgresql://fin@localhost:5432/fin_agent" \
--   -v ro_password="$FINANCIAL_QUERY_DB_PASSWORD" -f scripts/create_financial_query_readonly_role.sql
-- 不要把真实密码写入此文件或提交到仓库。

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fin_query_ro') THEN
        CREATE ROLE fin_query_ro LOGIN;
    END IF;
END
$$;

ALTER ROLE fin_query_ro PASSWORD :'ro_password';
ALTER ROLE fin_query_ro NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;

GRANT CONNECT ON DATABASE fin_agent TO fin_query_ro;
GRANT USAGE ON SCHEMA fin_core TO fin_query_ro;
REVOKE CREATE ON SCHEMA fin_core FROM fin_query_ro;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA fin_core FROM fin_query_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA fin_core TO fin_query_ro;

-- 后续由 fin 创建的新表默认也只授予查询权限。
ALTER DEFAULT PRIVILEGES FOR ROLE fin IN SCHEMA fin_core
    GRANT SELECT ON TABLES TO fin_query_ro;
