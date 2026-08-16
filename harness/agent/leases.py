"""进程内租约。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from harness.contracts.errors import AgentBusyError
from harness.session.types import new_id


@dataclass
class Lease:
    session_id: str
    owner_id: str
    token: str
    expires_at: datetime


class InMemoryLeaseStore:
    def __init__(self) -> None:
        self._leases: dict[str, Lease] = {}

    async def acquire(self, session_id: str, owner_id: str, *, ttl_seconds: float = 120) -> Lease:
        now = datetime.now(timezone.utc)
        current = self._leases.get(session_id)
        if current is not None and current.expires_at > now:
            raise AgentBusyError()
        lease = Lease(
            session_id=session_id,
            owner_id=owner_id,
            token=new_id(),
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self._leases[session_id] = lease
        return lease

    async def release(self, session_id: str, token: str) -> None:
        current = self._leases.get(session_id)
        if current is not None and current.token == token:
            self._leases.pop(session_id, None)
