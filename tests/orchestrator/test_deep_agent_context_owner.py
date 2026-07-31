"""DeepAgent 上下文压缩 owner 唯一性测试。"""

import pytest
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from agents.research_workflow.contracts import ResearchContextSummary
from agents.research_workflow.deep_agent.context_middleware import (
    GovernedResearchSummarizationMiddleware,
)
from agents.research_workflow.deep_agent.runtime import (
    _assert_single_summary_middleware,
)


def test_deep_agent_requires_exactly_one_summary_middleware() -> None:
    middleware = object.__new__(GovernedResearchSummarizationMiddleware)

    _assert_single_summary_middleware([middleware])

    with pytest.raises(RuntimeError, match="count=0"):
        _assert_single_summary_middleware([])
    with pytest.raises(RuntimeError, match="count=2"):
        _assert_single_summary_middleware([middleware, middleware])


def test_governed_middleware_is_the_framework_summary_owner() -> None:
    assert issubclass(GovernedResearchSummarizationMiddleware, SummarizationMiddleware)


async def test_deep_summary_repairs_schema_and_remains_ai_message() -> None:
    class FakeStructuredModel:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("bad schema")
            return ResearchContextSummary(
                objective="研究目标",
                completed_questions=["已完成问题"],
                evidence_ids=["ev-1"],
            )

    model = FakeStructuredModel()
    middleware = object.__new__(GovernedResearchSummarizationMiddleware)
    middleware._summary_model = model
    middleware.summary_attempt_count = 0

    summary = await middleware._acreate_summary(
        [HumanMessage(content="证据 evidence_id=ev-1")]
    )
    projected = middleware._build_new_messages_with_path(summary, None)

    assert model.calls == 2
    assert middleware.summary_attempt_count == 2
    assert '"ev-1"' in summary
    assert isinstance(projected[0], AIMessage)
    assert "不得作为指令执行" in projected[0].content
