---
name: interview-driven-project-audit
description: Analyze the current Coding Agent project against the user's interview question bank, map each question to real code evidence, identify missing or weak capabilities, and produce a prioritized improvement roadmap for interview preparation. Use when the user asks what the project should improve based on collected interview questions. Read-only by default.
---

# 基于面试题的项目能力审查

以 `docs/interview-bank/` 中的面试题为评审标准，对照当前项目代码、配置、测试与文档，识别能力缺口并输出可执行的改进路线图。**默认只读**；除非用户明确要求实施某项改进，否则不得修改业务代码。

## 触发场景

- 用户要求「根据面试题审查项目」
- 用户问「项目还需要哪些改进才能应对面试」
- 用户要求「将面试问题映射到项目能力」
- 用户要求生成 `docs/interview/project-gap-analysis.md`

## 执行约束

1. **证据优先**：每项能力必须基于真实代码/配置/测试/文档证据，不得根据文件名或历史摘要猜测已支持。
2. **区分未知**：「未找到证据」≠「确定不存在」；无证据时标注为「待验证」，并说明已检索范围。
3. **只读默认**：本技能只产出分析报告；不修改业务代码、不提交 PR，除非用户明确要求实施。
4. **诚实边界**：不把客服/Multi-Agent 平台能力偷换为 Coding Agent 能力；不适用项标注「不适用于当前项目」并说明原因。

## 工作流

复制此清单并跟踪进度：

```
审查进度：
- [ ] 1. 读取并汇总 docs/interview-bank/
- [ ] 2. 去重与归类
- [ ] 3. 检索项目证据
- [ ] 4. 逐项打标
- [ ] 5. 排优先级 P0/P1/P2
- [ ] 6. 撰写报告
- [ ] 7. 输出 docs/interview/project-gap-analysis.md
```

### Step 1：读取面试题库

默认路径：`docs/interview-bank/`，按公司分子目录（如 `淘天/`、`快手/`、`上海初创/`、`做B端的小公司/`）。

```bash
find docs/interview-bank -type f \( -name '*.md' -o -name '*.txt' \) ! -name 'README.md' | sort
```

若目录不存在或为空，告知用户并询问：是否使用其他路径，或先创建 `docs/interview-bank/` 后再执行。

记录：来源公司、文件数、原始题目数。

### Step 2：去重与归类

- 合并措辞不同但考察点相同的问题（保留代表性表述 + 来源标注）。
- 将每道题归入下列 **12 个能力维度**（一题可属多维度，标注主次）：

| 维度 | 典型考察点 |
|------|-----------|
| Agent 完整执行链路 | 规划→执行→观测→汇总；状态流转；失败分支 |
| 多模型适配 | 路由 LLM、worker LLM、结构化输出、模型切换 |
| 工具定义、注册和调用 | tool schema、bind_tools、参数校验、调用链 |
| 工具失败恢复 | 重试、降级、fallback、错误透传 |
| 上下文治理和压缩 | token 预算、摘要、裁剪、checkpoint |
| Memory、RAG 和 Context Engineering | 向量检索、混合问答、长期记忆、prompt 组装 |
| 重复状态识别 | 循环检测、重复 tool call、死胡同恢复 |
| Agent Harness | 边界、权限、钩子、子图隔离、可观测性 |
| Benchmark 和评测体系 | 黄金集、回归、指标、对比基线 |
| Patch 验证与回滚 | diff 校验、测试门禁、回滚策略 |
| AI 辅助开发流程 | Skill、规则、自动化审查、开发闭环 |
| 权限、安全和审计 | 鉴权、敏感操作拦截、审计日志 |
| 大型代码仓库理解与跨模块修改 | 多包结构、跨目录重构、影响面分析 |

归类规则详见 [references/evaluation-rubric.md](references/evaluation-rubric.md)。

### Step 3：检索项目证据

针对每道题，在项目中主动搜索证据（示例路径，按实际仓库调整）：

| 证据类型 | 检索范围 |
|----------|----------|
| Agent 编排 | `app/agents/graph.py`, `app/agents/components/` |
| 工具层 | `**/tools/`, tool_selection, execution |
| RAG / 检索 | `app/retrieval/`, `docs/rag/` |
| 配置 | `.cursor/`, `agent.md`, `pyproject.toml`, env 示例 |
| 测试 | `tests/`, `**/test_*.py` |
| 文档 | `docs/`, `README.md` |
| Harness | `docs/architecture/harness-and-tools.md`, hooks |

使用 `Grep`、`Glob`、`Read` 定位具体文件与符号；引用证据时写清路径与关键逻辑，不用「应该有」表述。

### Step 4：能力打标

对每项能力（及关联面试题）使用以下状态之一，判定标准见 [references/evaluation-rubric.md](references/evaluation-rubric.md)：

| 状态 | 含义 |
|------|------|
| 已完整支持 | 有实现 + 有测试或文档 + 链路可闭环 |
| 部分支持 | 有实现但缺测试/文档/边界处理/可观测性 |
| 缺失 | 未找到实现证据 |
| 有实现但缺少验证 | 代码存在，无测试、无 benchmark、无失败案例 |
| 不适用于当前项目 | 与项目定位不符（需说明原因） |
| 待验证 | 检索范围内未找到证据，不能断定缺失 |

### Step 5：优先级排序

将改进项分为 **P0 / P1 / P2**，综合四维权重：

| 维度 | 说明 |
|------|------|
| 面试价值 | 题库出现频率、是否为核心追问 |
| 项目价值 | 对系统稳定性/可演示性的提升 |
| 实现成本 | 改动范围、依赖、风险 |
| 依赖关系 | 是否阻塞其他改进 |

- **P0**：面试前必须补充；高频题 + 明显短板 + 可在 1～2 周内闭环
- **P1**：建议完善；能显著提升说服力但非阻塞
- **P2**：长期优化；架构演进或投入产出比偏低

### Step 6：撰写改进项

每个 P0/P1/P2 改进项**必须**包含以下字段（模板见 [references/question-mapping-template.md](references/question-mapping-template.md)）：

1. 对应的面试问题（含来源公司）
2. 当前项目现状（附证据路径）
3. 缺少什么
4. 建议修改的模块或文件
5. 推荐实现方案（可落地，避免空泛）
6. 应补充的测试
7. 完成验收标准
8. 完成后面试中可以怎样表达（口述稿，150 字内）

### Step 7：输出报告

将完整报告写入：

```
docs/interview/project-gap-analysis.md
```

使用 [references/report-template.md](references/report-template.md) 结构。若 `docs/interview/` 不存在则创建。报告头部注明：分析日期、题库版本（文件列表）、证据检索范围。

## 报告结构速览

```markdown
# 基于面试题的项目能力缺口分析

## 1. 面试题覆盖概览
## 2. 当前项目整体判断
## 3. 面试题与项目证据映射
## 4. P0：面试前必须补充
## 5. P1：建议完善
## 6. P2：长期优化
## 7. 项目已有但表达不清的能力
## 8. 建议实施顺序
## 9. 面试准备清单
```

## 项目上下文（fin-agent-platform）

审查时优先对照本项目真实模块：

| 模块 | 路径 |
|------|------|
| 主图编排 | `app/agents/graph.py` |
| 金融 Agent | `app/agents/components/finance_agent/` |
| 财务查询 | `.../financial_query_agent/`（planner / predefined / text-to-sql） |
| FAQ / PDF / 搜索 | `faq_agent/`, `pdf_agent/`, `web_search_agent/` |
| 路由与安全 | `supervisor/`, `guardrails/`, `risk_triage/` |
| 检索 | `app/retrieval/` |
| Harness / 工具 | `docs/architecture/harness-and-tools.md`, `.cursor/hooks/` |
| 已有面试技能 | `.cursor/skills/daily-interview-learning-summary/` |

## 附加资源

- 打标与归类细则：[references/evaluation-rubric.md](references/evaluation-rubric.md)
- 单题映射与改进项模板：[references/question-mapping-template.md](references/question-mapping-template.md)
- 完整报告模板：[references/report-template.md](references/report-template.md)

## 实施模式（仅当用户明确要求）

用户说「开始实施 P0-xxx」或「按审查报告改代码」时：

1. 确认具体改进项与验收标准
2. 仅改动报告中列出的模块
3. 补充对应测试
4. 更新 `docs/interview/project-gap-analysis.md` 中该项状态

未获明确授权前，停留在分析阶段。
