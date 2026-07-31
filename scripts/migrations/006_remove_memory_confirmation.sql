-- 移除候选确认流程；历史 pending 偏好不再等待用户操作。
UPDATE app.memory_records
SET
    status = 'rejected',
    consent_status = 'denied',
    updated_at = now()
WHERE memory_type = 'preference'
  AND status = 'pending';
