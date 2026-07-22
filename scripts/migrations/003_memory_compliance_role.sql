-- 管理员合规角色。
ALTER TABLE app.users ADD COLUMN IF NOT EXISTS role VARCHAR(32);
UPDATE app.users SET role = 'user' WHERE role IS NULL;
ALTER TABLE app.users ALTER COLUMN role SET DEFAULT 'user';
ALTER TABLE app.users ALTER COLUMN role SET NOT NULL;
CREATE INDEX IF NOT EXISTS ix_users_role ON app.users (role);
