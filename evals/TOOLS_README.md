# 工具边界与稳定性评测集

两套独立 JSONL：`datasets/fin-harness-tools-smoke-v1.jsonl`（3条调通集）和 `datasets/fin-harness-tools-v1.jsonl`（100条全面集）。完整用例目录见 [TOOL_CASES.md](TOOL_CASES.md)。

这是测试规格与数据交付，不是已运行的成功率报告。当前 `run_core_eval` 和 `HarnessEvalAdapter` 不会消费本集合的 fixture/assertions，不可直接拿它们运行后宣称测试通过。本次提供离线结构校验、本地导出和显式 LangSmith 上传入口；场景驱动、断言执行及真实服务评测须另行接入。

## 3 条先调通

| ID | 链路 | 固定期望 |
|---|---|---|
| tool-smoke-001 | 财务计算 | 100万元到120万元，同比增长20% |
| tool-smoke-002 | 问财 → 证据 → 计算 → 回答 | 2023年100万元、2024年120万元、同比20% |
| tool-smoke-003 | 财报检索 + 联网搜索 → 分别引用 | 年报第10页收入120万元；测试公告链接支持新品发布 |

调通顺序：先验证加载/导出；再给真实 Harness 注入 Fake provider，以真实模型或 Fake LLM 运行；最后连接 LangSmith 核对父子 trace、最终答案与断言 feedback。四种业务工具都被覆盖，但3条不足以衡量稳定性。smoke 基于全面集扩展，不是独立盲测集，不能与全面集相加当103条独立统计样本。

## 100 条覆盖

| 范围 | 数量 | 重点 |
|---|---:|---|
| tool-001～020 财务计算 | 20 | 同比、负基数、零基数、单位、百分点、CAGR、批量、证据引用、非有限值、精度 |
| tool-021～040 问财 | 20 | 实体、市场、时间、指标口径、筛选边界、空值、分页、429、超时、401、重复与多轮 |
| tool-041～060 财报 | 20 | 文档/年份/口径锁定、表头单位、跨页、重述、分部、引用、注入、OCR、矛盾证据 |
| tool-061～080 联网 | 20 | 来源、时区、时效、域名、注入、正文缺失、故障、矛盾、去重、隐私、内网链接 |
| tool-081～100 Registry/运行时/MCP | 20 | JSON/schema、注册冲突、MCP发现/业务错误/断线、审批、取消、循环、并发、租约、恢复、隔离、SSE |

## 数据契约及执行要求

- `inputs`：只含用户 query 或用户 turns；不得将金答案、故障配置或 fixture 注入模型提示词。
- `fixture`：测试驱动专用配置。全部是合成数据，测试公司/股票代码不是现实证券，`.example` URL 是假HTTP服务地址；禁止真实外网请求。时间固定于北京时间2026年10月5日10:00。
- `fixture.setup` 是描述性场景规格，不是现有 provider 的原生响应 schema。驱动须将其转换为真实工具所需的返回结构，注册可解析的 evidence ID，保留公司、期间、单位及locator。不允许把测试期望直接伪装成Agent答案。
- 有 `responses`/`mcp_sequence` 的场景按顺序返回结果或注入故障；超出次数后驱动应失败，而不是无限复用最后一次成功响应。无列表的场景使用静态返回。固定时钟不应冻结用于 deadline 的单调时钟。
- `outputs.required_facts` 是可选结构化期望；为空不代表免评分。完整判分要求在 `assertions` 与 `global_gates`。
- `metadata.execution_mode` 决定测试入口：`agent` 走真实 Harness 问答；`tool_contract` 直接调用 schema/pipeline/capability；`runtime` 使用 Fake LLM、可控工具及必要的临时数据库/HTTP/SSE服务；`answer_evaluation` 测试评估器能否识别给定错误答案。
- registry、schema、策略参数使用隔离测试配置，不改变生产全局状态。每条独立会话/用户/缓存命名空间；多轮例外必须在同一用例中延续状态。
- 缺少fixture驱动、依赖或语义judge时记为 `not_evaluated`，不得算通过。某功能当前尚未实现时，应报告失败/能力缺口，不调整金标准以迁就实现。

## 断言与指标

每条断言输出 `case_id / assertion_id / status(pass,fail,not_evaluated) / reason / observed / trace_id`。全部硬断言与全局门槛通过才算 case pass。数值使用声明的 tolerance；未声明时精确核验业务单位下的期望值，不用模糊字符串包含来判断事实。公司、期间等必要身份缺失不能判正确。

`trace_or_code` 由确定性代码检查请求计数、事件、handler spy、并发峰值、时间线、数据库状态。`trace_and_answer` 同时核验工具过程与最终文本中的事实、口径、来源；语义部分可人工审核或使用经过人工校准的judge。不能仅检查工具中存在正确facts就给最终答案满分。

分别报告：

1. 场景通过率：全面集及各能力/执行层的通过比例；另列未评数量和评估覆盖率。
2. 首次调用成功率：首次成功的逻辑调用 / 已实际发起的逻辑调用；事先被参数/权限校验拒绝的请求单独记录拒绝正确率。
3. 有界重试后成功率：预算内最终成功的逻辑调用 / 已实际发起的逻辑调用；同时记录实际底层请求总数。
4. 重复调用率：无新信息、无必要重试理由的重复实际请求 / 实际请求数。以规范化参数、身份、来源与时效共同确定请求等价性。
5. 任务完成率、无证据结论率、来源违规率、p50/p95时延、取消延迟、在途峰值。

本集刻意包含失败注入：401后正确停止是“场景通过”，不是“工具调用成功”。故障集成功率、真实供应商可用率、业务任务完成率必须分开。100条按风险覆盖分布，不代表生产流量，不能直接宣称生产XX%成功率。

建议关键场景重复5次，报告平均通过率、5次全通过的样本比例、样本数及未评数。相同场景多次运行存在相关性，不能把它们作为完全独立样本夸大统计置信度。

## 本地校验与 LangSmith

```bash
.venv/bin/python -m evals.tool_dataset --source smoke
.venv/bin/python -m evals.tool_dataset --source full
.venv/bin/python -m evals.tool_dataset --source smoke --export /tmp/tool-smoke-langsmith.jsonl
```

运行 3 条真实 Loop smoke：

```bash
pytest tests/tool_scenarios -m smoke
```

运行 100 条参数化 Scheduler/Pipeline 场景：

```bash
pytest tests/tool_scenarios/test_full_dataset.py -m full
```

全量入口会执行真实参数解析、Pipeline、错误归一化、重试预算、Scheduler
和事件配对，并把未接入的最终答案、来源绑定、MCP 服务、跨进程数据库等断言明确记为
`not_evaluated`。pytest 的 passed 只表示已执行的结构断言通过；是否完成整条业务场景应读取
报告中的 `evaluation_status`、`evaluation_coverage` 和 `scenario_pass_rate`。当前产品会在
粘连 JSON 与非法尾缀两个契约用例上失败，因此命令返回非零是有效缺陷信号。

## 真实模型 + 假工具行为评测

这一层使用真实 `DeepSeekAdapter` 和真实 Agent Loop，但所有问财、财报、搜索和计算工具
都由脚本化 provider 返回合成结果。fixture 只交给假工具和评估器，不会拼进被测模型的
prompt。默认选20条校准集，并使用一次独立结构化 judge 检查最终事实、行为、来源和编造。

测试必须显式开启，避免普通 pytest 意外消耗模型额度：

```bash
pytest tests/tool_scenarios/test_model_behavior.py \
  -m model_behavior \
  --run-model-behavior \
  --model-repetitions 1 \
  --tb=short
```

关键用例建议重复5次：

```bash
pytest tests/tool_scenarios/test_model_behavior.py \
  -m model_behavior \
  --run-model-behavior \
  --model-repetitions 5 \
  --tb=short
```

运行指定用例：

```bash
pytest tests/tool_scenarios/test_model_behavior.py \
  -m model_behavior \
  --run-model-behavior \
  --model-case-id tool-031 \
  --model-case-id tool-050 \
  --model-repetitions 5
```

全部70条 `agent` 用例：

```bash
pytest tests/tool_scenarios/test_model_behavior.py \
  -m model_behavior \
  --run-model-behavior \
  --model-all \
  --model-repetitions 5
```

最后一条会产生最多350次Agent任务以及相应judge调用，成本和运行时间较高。先跑单条，
再跑20条校准集，稳定后再全量。结果写入
`.pytest_cache/tool_scenarios/latest-model-behavior.json`，与模块测试报告分开。

配置凭据后，显式上传（本次交付没有执行上传）：

```bash
.venv/bin/python -m evals.tool_dataset --source smoke --upload
.venv/bin/python -m evals.tool_dataset --source full --upload
```

LangSmith 示例保留 `inputs`、`outputs`，fixture 放在 `metadata.fixture`，并带稳定 `case_id` 和 `case_sha256`。运行器通过case_id从本地读取fixture，不把metadata发送给模型。上传跳过相同内容；同ID内容变化时拒绝静默覆盖，使用 `--dataset-name fin-harness-tools-v2` 发布新版本。

实验记录模型版本、prompt hash、commit、fixture版本、工具schema hash、策略参数和judge版本。每种 execution_mode 独立执行、统一上传feedback；禁止把所有条目交给一个问答target。线上测试另用真实供应商探针集，本集合的合成数据不能用于确认真实公司财务事实。
