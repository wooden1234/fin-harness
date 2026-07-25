---
name: daily-interview-learning-summary
description: 总结当天在 fin-agent-platform 项目中的开发、调试、排障与架构学习，整理成面试复盘并落盘到 docs/diary/YYYY-MM-DD.md。Use when the user asks to 总结今天的收获、整理今天学到的东西、把今天的修改整理成面试能讲的内容、总结 fin-agent-platform 项目经验、沉淀面试话术、或执行 /daily-interview-learning-summary。
---

# 今日面试复盘总结（fin-agent-platform）

## 目标

1. 不只是罗列今天做了什么，而是提炼「我学到了什么、解决了什么问题、用了什么方法、体现了什么能力」。
2. 输出内容要适合后续面试复盘和自我介绍使用。
3. 重点总结技术理解、架构思路、排查过程、工程能力、失败经验和可讲故事。
4. 区分事实、理解和待补充内容，不编造不存在的项目经历。
5. 语言要真实、具体、可落地，避免空话和套话。

## 执行步骤

1. **收集今日上下文**（在 `fin-agent-platform` 仓库根目录执行）：
   - 当前对话及同一会话中的历史消息
   - 今日 git 变更：
     ```bash
     git log --since="midnight" --oneline
     git diff --stat
     git diff
     ```
   - 今日改动的模块路径（对照下方项目地图）
   - 用户明确提到的任务、问题、决策
   - 若信息不足，向用户确认关键任务后再写，不猜测

2. **提炼而非流水账**：每条收获必须能对应到具体任务、问题或代码实践，尽量引用真实模块/文件路径。

3. **标注可信度**（在正文中自然体现，无需单独章节）：
   - **事实**：确实发生或完成的
   - **理解**：基于今日实践形成的认知
   - **待补充**：还不确定、需进一步验证或深化的点

4. **按下方模板撰写 Markdown**，控制篇幅，不写成长篇日报。

5. **落盘到仓库**（默认必须执行，除非用户明确只要聊天内预览）：
   - 目标路径：`docs/diary/YYYY-MM-DD.md`（日期取用户时区当天，格式 `YYYY-MM-DD`）
   - 若 `docs/diary/` 不存在则创建
   - 若当日文件已存在：在文末追加 `---` 分隔的新一节，标题带时间戳；**不要**无声覆盖全文（用户说「覆盖/重写今日日记」时除外）
   - 同步更新 `docs/diary/README.md` 索引表：新增一行「日期 | 主题」（主题取 §1 一句话）
   - 聊天回复中告知用户落盘路径，并给 1～2 句摘要

## 文档落盘位置

| 项 | 约定 |
|----|------|
| 日记目录 | `docs/diary/` |
| 单日文件 | `docs/diary/YYYY-MM-DD.md` |
| 目录索引 | `docs/diary/README.md` |
| 关联缺口报告 | `docs/interview/project-gap-analysis.md`（若当日做了 audit，在日记「关联产出」中链接） |
| 文档索引入口 | `docs/README.md` 已登记 `diary/` |

落盘文件在模板正文末尾可增加可选块：

```markdown
## 关联产出

| 文件 | 说明 |
|------|------|
| `docs/interview/project-gap-analysis.md` | （若有）当日缺口分析 |
| `app/...` | （若有）当日主要改动模块 |
```

## 项目上下文

复盘时优先用本项目术语，把经历锚定到真实模块：

| 模块 | 路径 | 面试可讲点 |
|------|------|-----------|
| 主图编排 | `app/agents/graph.py` | LangGraph 多节点流水线、guardrails 短路、子图接入 |
| 金融 Agent 子图 | `app/agents/components/finance_agent/` | planner → workers → join → summarize 并行检索与汇总 |
| 财务查询 Agent | `.../financial_query_agent/` | predefined 白名单/SQL、text-to-sql、语义解析 |
| FAQ / PDF / 搜索 | `faq_agent/`, `pdf_agent/`, `web_search_agent/` | 多 worker 隔离与 fan-in |
| 路由与安全 | `supervisor/`, `guardrails/`, `risk_triage/` | 意图路由、合规拦截 |
| API 层 | `app/api/` | FastAPI 接口、流式进度 |
| 检索 | `app/retrieval/` | RAG 检索链路 |

项目定位：**金融 Multi-Agent 智能客服**（LangGraph + FastAPI）。详细架构见 [reference.md](reference.md)，Coding Agent 面试题库见 [interview-questions.md](interview-questions.md)。

## 执行模式

- **日常复盘**（默认）：走下方 6 节模板；§4 从 8 道主问中选 2～3 道。
- **面试刷题**：用户说「按 Coding Agent 题库复盘 / 刷题」时，按 [interview-questions.md](interview-questions.md) 分组输出口述稿；仍标注事实/理解/待补充，不编造。

## 整理维度

执行时覆盖以下维度（融入各章节，不必逐条标题列出）：

- 今日主要任务
- 关键技术点
- 遇到的问题
- 解决思路
- 最终收获
- 可以在面试中怎么讲
- 面试官可能追问的问题
- 明天需要继续补强的内容

## 输出模板

```markdown
# 今日面试复盘总结

> 项目：fin-agent-platform · [日期]

## 1. 今日主要内容

简要总结今天主要做了什么。

## 2. 核心收获

提炼 3～5 条真正有价值的技术或工程收获。

## 3. 可用于面试的项目表达

把今天的经历整理成面试中可以讲的一段话。

## 4. 面试官可能追问

从 [interview-questions.md](interview-questions.md) 的 8 道主问中，按今日改动选 2～3 道，每道附 2～3 句回答思路；可带 1 条相关追问。未触及的维度标「待补强」，不编造。

格式示例：
- **Q3 上下文压缩**（本项目）：…
- **Q2 工具失败恢复**（方法论）：…

## 5. 暴露出的短板

总结今天发现自己还不熟、不够深入或容易被问倒的地方。

## 6. 明日补强建议

给出后续学习或项目完善建议。
```

## 输出要求

- 简短、真实、具体。
- 不要写成长篇日报。
- 不要编造没有发生的内容。
- 不要只写流水账。
- 每条收获都要能对应到具体任务、问题或代码实践。
- 尽量转化成面试可讲的表达。
- 提及模块时使用 `app/...` 真实路径，便于后续查阅。
- **默认写入** `docs/diary/YYYY-MM-DD.md`，并在回复中返回该路径。

## 章节写作指引

**§1 今日主要内容**：1～3 句话，说清在 fin-agent-platform 哪个模块做了什么。

**§2 核心收获**：每条格式建议为「收获 → 对应场景/问题 → 体现的能力」。优先写 Multi-Agent 编排、LangGraph、金融查询链路、RAG、工程化调试等可讲故事的点。

**§3 可用于面试的项目表达**：用 STAR 或「背景 → 挑战 → 行动 → 结果」组织成 1 段可直接口述的话（150～250 字），第一人称，项目名用「金融 Multi-Agent 智能客服平台」。

**§4 面试官可能追问**：

1. 先读 [interview-questions.md](interview-questions.md)，按「选题速查」表匹配今日 git 变更。
2. 从 8 道主问（Q1–Q8）中选 **2～3 道**，宁少勿多；每道标注 **本项目 / 个人实践 / 方法论**。
3. 每道写：主问题 + 2～3 句回答思路 + 项目锚点（若有）；可选 1 条常见追问。
4. 今日未触及的主问：不在 §4 硬答，写入 §5「待补强」。
5. 选题参考：改了 compressor → Q3；worker/tool → Q2；graph/planner → Q1；retrieval → Q7；调试改错 → Q6；个人 AI 流程 → Q8。
6. **诚实边界**：禁止把客服 Agent 说成 Coding Agent；禁止编造未实现的记忆/评测/回滚系统。

**§5 暴露出的短板**：基于今日真实卡点，不写泛泛的「还需加强算法」。

**§6 明日补强建议**：具体、可执行，与 §5 短板对应，可指向本项目待完善模块或文档。

## 落盘检查清单

执行结束时自检：

```
- [ ] 已写入 docs/diary/YYYY-MM-DD.md
- [ ] 已更新 docs/diary/README.md 索引（新日期才加行）
- [ ] 聊天回复已注明落盘路径
- [ ] 未编造未发生的实现或评测
```
