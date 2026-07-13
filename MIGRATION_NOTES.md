# fin-harness 结构化迁移说明

`fin-harness` 不是对 `fin-agent-platform` 的机械复制，而是按 Harness 架构重新组织现有内容。

## 已迁入的现有内容

- `app/agents/`：保留现有 LangGraph 主图、finance 子图和各业务 Agent。
- `app/retrieval/`：保留现有 FAQ/PDF RAG 检索基础设施。
- `app/api/`：保留现有 HTTP/SSE 入口，后续改为调用 `app.harness.runner`。
- `app/core/`：保留配置、数据库、日志、安全等基础设施。
- `app/models/`、`app/schemas/`、`app/services/`：保留现有数据模型和服务层。
- `scripts/`、`tests/`、`docs/`、`knowledge/`：保留评测、脚本、文档和知识库内容。

## 新增的 Harness 分层

- `app/harness/`：运行治理层，统一处理上下文、策略、注册、审计、错误和图调用。
- `app/tools/`：原子工具层，包装现有检索、SQL、联网搜索等动作。
- `app/skills/`：业务能力层，编排 tools 或复用现有 agent workflow。
- `app/mcp/`：MCP 接入层，后续统一对接外部系统。
- `app/evidence/`：证据和引用治理层。
- `app/compliance/`：合规规则和输出审查层。
- `app/audit/`：审计事件、回放和脱敏层。
- `app/evals/`：上线评测、回归和 replay 评估层。

## 迁移原则

1. `Agent` 负责判断、规划、审查，不直接绕过工具访问外部系统。
2. `Skill` 负责编排多步业务流程，不重复实现底层检索、SQL、联网逻辑。
3. `Tool` 只做原子动作，并经过权限、超时、审计控制。
4. `MCP` 只做外部系统连接，不承载业务逻辑。
5. `Harness` 负责运行治理，不写具体金融业务问答逻辑。

## 下一步建议

1. 将 `app/api/agent.py` 的图调用改为 `app.harness.runner.run_agent`。
2. 将 `faq_agent`、`pdf_agent`、`financial_query_agent` 逐步改为调用对应 skill。
3. 将现有 Tavily 搜索、RAG 检索、财务 SQL 查询逐步收敛到 `app/tools/`。
4. 在 `evidence` 和 `compliance` 通过后再输出 `final_answer`。
