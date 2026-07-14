# fin-harness

金融 Multi-Agent 智能客服平台。基于 LangGraph 编排多 Agent 协作，支持 FAQ 问答、PDF 研报检索、财务数据查询与联网搜索，并提供合规审查、证据引用与审计能力。

## 功能概览

- **Supervisor 路由**：根据用户意图分发至 FAQ、PDF、财务查数、联网搜索等子 Agent
- **财务查数**：预定义 SQL 模板 + Text-to-SQL 双路径，覆盖年报指标与结构化查询
- **RAG 检索**：LlamaIndex + pgvector 混合检索，支持 FAQ、宏观、年报、研报等多集合
- **Harness 治理**：统一运行上下文、策略、工具注册、合规审查与审计回放
- **Web 前端**：React + Vite 聊天界面，SSE 流式输出 Agent 执行步骤

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI · SQLAlchemy · PostgreSQL · Redis |
| Agent | LangGraph · LangChain · DeepSeek |
| 检索 | LlamaIndex · pgvector · BM25 |
| 前端 | React · TypeScript · Vite · Tailwind CSS |

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### 1. 克隆与依赖

```bash
git clone https://github.com/wooden1234/fin-harness.git
cd fin-harness

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、QWEN_API_KEY 等
```

### 2. 启动基础设施

```bash
docker compose up -d
python scripts/init_db.py
python scripts/setup_langgraph_checkpoint.py
```

默认连接信息见 `.env.example`（PostgreSQL `fin:fin@localhost:5432/fin_agent`，Redis `localhost:6379`）。

### 3. 启动后端

```bash
cd app/backend
PYTHONPATH=../.. uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

- 健康检查：<http://127.0.0.1:8010/health>
- API 文档：<http://127.0.0.1:8010/docs>

### 4. 启动前端（可选）

```bash
cd app/frontend
npm install
npm run dev
```

前端默认运行在 <http://127.0.0.1:5173>。

## 项目结构

```
fin-harness/
├── agents/          # LangGraph Agent 图（Supervisor、FAQ、PDF、财务查数等）
├── app/
│   ├── backend/     # FastAPI 后端（API、模型、服务）
│   └── frontend/    # React 前端
├── retrieval/       # RAG 索引与检索
├── harness/         # 运行治理层
├── tools/           # 原子工具（SQL、检索、联网等）
├── skills/          # 业务能力编排
├── mcp/             # MCP 外部系统接入
├── evidence/        # 证据与引用
├── compliance/      # 合规规则与审查
├── audit/           # 审计与回放
├── evals/           # 评测、回归与评测脚本
├── scripts/         # 初始化与数据导入脚本
└── tests/           # 单元与集成测试
```

## 测试

```bash
pytest
```

需要 LLM / Embedding API Key 的集成测试会自动跳过（见 `tests/conftest.py`）。

## 环境变量

关键配置项（完整列表见 `.env.example`）：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek LLM API Key |
| `QWEN_API_KEY` | DashScope Embedding API Key |
| `DATABASE_URL` | PostgreSQL 异步连接串 |
| `PGVECTOR_DATABASE_URL` | pgvector 连接串 |
| `LANGGRAPH_CHECKPOINT_URL` | LangGraph 状态持久化 |
| `SECRET_KEY` | JWT 签名密钥 |

> `.env` 已在 `.gitignore` 中，请勿提交至仓库。

## License

Private project — all rights reserved.
