"""Main DeepAgent 的无工具收尾中间件。"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_stream_writer

from agents.main_deep_agent.middleware.budget import MainAgentBudgetController


class MainAgentFailureFinalizationMiddleware(AgentMiddleware):
    """进入收尾后仅移除业务工具，保留结构化响应工具。"""

    def __init__(
        self,
        budget: MainAgentBudgetController,
        *,
        business_tool_names: set[str] | None = None,
    ) -> None:
        self.budget = budget
        self.business_tool_names = set(business_tool_names or set())
        self._event_emitted = False

    async def awrap_model_call(self, request, handler):
        if self.budget.stop_new_tools:
            if not self._event_emitted:
                self._event_emitted = True
                try:
                    get_stream_writer()(
                        {
                            "kind": "main_finalization",
                            "status": "running",
                        }
                    )
                except RuntimeError:
                    pass
            tools = [
                tool
                for tool in list(getattr(request, "tools", []) or [])
                if getattr(tool, "name", "") not in self.business_tool_names
            ]
            current_system = str(getattr(request, "system_message", "") or "")
            return await handler(
                request.override(
                    tools=tools,
                    system_message=(
                        current_system
                        + "\n现在进入无工具收尾阶段。必须使用 MainAgentResponse 结构，"
                        "只基于已取得 Evidence 成稿；不得发起新工具调用或用模型记忆补事实。"
                    ),
                )
            )
        return await handler(request)


__all__ = ["MainAgentFailureFinalizationMiddleware"]

