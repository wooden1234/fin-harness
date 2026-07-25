# fin-agent-platform 架构速查

供 `daily-interview-learning-summary` 复盘时对照模块，无需全文背诵。

## 技术栈

- Python · FastAPI · LangGraph · uvicorn
- 本地启动：`uvicorn app.main:app --reload --port 8000`

## 主图（`app/agents/graph.py`）

```
START → guardrails → context_compressor → supervisor → risk_triage
  → general_agent | plan_agent(finance_agent 子图) → final_answer → END
```

- **guardrails**：输入校验，可短路至 final_answer
- **context_compressor**：上下文压缩
- **supervisor**：意图分析与路由
- **risk_triage**：风险分级
- **plan_agent**：finance_agent 子图（对外单一节点）

## finance_agent 子图（`app/agents/components/finance_agent/graph.py`）

```
START → supervisor → [faq_agent | pdf_agent | financial_query_agent | web_search_agent]
  → join → summarize → END
```

- **workers**：`isolate_worker_node` 隔离各检索 worker
- **join**：fan-in 合并多路结果
- **summarize**：汇总生成中间答案

## financial_query_agent

路径：`app/agents/components/finance_agent/financial_query_agent/`

- **predefined/**：白名单指标、语义解析、SQL 构建、registry
- **text_to_sql/**：自然语言转 SQL 执行链路
- **workflows/**：编排 predefined 与 text-to-sql 流程

## 其他关键目录

| 目录 | 职责 |
|------|------|
| `app/api/` | HTTP 接口、agent 调用与进度 |
| `app/core/` | 配置、LLM 客户端等基础设施 |
| `app/models/` | 数据模型 |
| `app/retrieval/` | 检索相关 |
| `app/schemas/` | Pydantic 模式 |
| `app/services/` | 业务服务层 |

## 面试叙事角度（可选）

- **Multi-Agent 编排**：主图 + 子图分层，职责边界清晰
- **并行检索 + Join**：多 worker 独立执行后汇总，降低单点失败影响
- **金融查询双路径**：predefined 白名单（可控、低延迟）vs text-to-sql（灵活、需校验）
- **安全链路**：guardrails + risk_triage 在生成前拦截
- **工程化**：FastAPI 暴露能力、LangGraph checkpoint 支持状态恢复（若今日涉及）
