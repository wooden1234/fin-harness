"""Agent 运行过程与上下文指标查询 API。"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import get_current_user
from app.models.identity.user import User
from app.schemas.agent_events import AgentRunEventPage, AgentRunEventRead
from app.services.agent.context_event_service import ContextEventService

router = APIRouter(prefix="/agent/runs", tags=["agent-events"])


def _platform_admin(user: User) -> bool:
    return str(user.role) == "platform_admin"


@router.get("/{run_id}/events", response_model=AgentRunEventPage)
async def list_agent_run_events(
    run_id: str,
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
) -> AgentRunEventPage:
    try:
        rows = await ContextEventService.list_events(
            run_id=run_id,
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            platform_admin=_platform_admin(current_user),
            after_id=after_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    items = [AgentRunEventRead.model_validate(item) for item in rows]
    return AgentRunEventPage(
        items=items,
        next_after_id=items[-1].id if len(items) == limit else None,
    )


@router.get("/{run_id}/context-metrics")
async def get_agent_run_context_metrics(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    try:
        return await ContextEventService.metrics(
            run_id=run_id,
            tenant_id=current_user.tenant_id,
            user_id=current_user.id,
            platform_admin=_platform_admin(current_user),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
