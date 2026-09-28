"""PostgreSQL-backed cross-process Session leases."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from harness.agent.leases import Lease
from harness.contracts.errors import AgentBusyError
from harness.session.types import new_id


class PostgresLeaseStore:
    async def acquire(self, session_id: str, owner_id: str, *, ttl_seconds: float = 120) -> Lease:
        from app.core.database import AsyncSessionLocal

        token = new_id()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text(
                    "INSERT INTO app.agent_session_leases "
                    "(session_id, owner_id, token, expires_at, fencing_token, updated_at) "
                    "VALUES (:session_id, :owner_id, :token, :expires_at, 1, NOW()) "
                    "ON CONFLICT (session_id) DO UPDATE SET "
                    "owner_id = EXCLUDED.owner_id, token = EXCLUDED.token, "
                    "expires_at = EXCLUDED.expires_at, "
                    "fencing_token = app.agent_session_leases.fencing_token + 1, updated_at = NOW() "
                    "WHERE app.agent_session_leases.expires_at <= :now "
                    "RETURNING token"
                ),
                {
                    "session_id": session_id,
                    "owner_id": owner_id,
                    "token": token,
                    "expires_at": expires_at,
                    "now": now,
                },
            )).first()
            if row is None:
                await db.rollback()
                raise AgentBusyError()
            await db.commit()
        return Lease(session_id=session_id, owner_id=owner_id, token=token, expires_at=expires_at)

    async def release(self, session_id: str, token: str) -> None:
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await db.execute(
                text(
                    "DELETE FROM app.agent_session_leases "
                    "WHERE session_id = :session_id AND token = :token"
                ),
                {"session_id": session_id, "token": token},
            )
            await db.commit()


__all__ = ["PostgresLeaseStore"]
