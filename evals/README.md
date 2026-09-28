# fin-harness 端到端评测集

`datasets/fin-harness-core-v1.jsonl` 是面向真实用户任务的 50 条端到端核心集。问题先按用户意图设计，再映射到稳定能力域；不以某个 Python 函数名或供应商工具名作为产品契约。

## 设计原则

- `inputs` 只描述用户实际会提供的信息；多轮样本使用 `turns`。
- `outputs.required_facts` 用于可复现的固定事实判分，数字同时保留单位和来源定位。
- `outputs.required_behaviors` 用于实时数据、澄清、合规和无答案场景，避免编造静态金答案。
- `outputs.forbidden_behaviors` 是硬门槛，例如串公司、串年份、伪造引用和给出确定性买卖指令。
- 配套 routing 文件定义用户意图对应的稳定能力域和来源策略。能力域是企业契约，具体工具只是实现。
- `metadata.diagnostic_capabilities` 仅用于定位失败，不应直接作为 pass/fail 条件。
- 评测最终答案，不要求逐字匹配；数值与口径采用确定性评估，语义完整性采用人工或 LLM judge。

## 题型分布

| 题型 | 数量 |
|---|---:|
| 单任务：财务事实、筛选、公告、评级和公司信息 | 16 |
| 比较、计算与口径 | 8 |
| 多证据综合与来源归因 | 7 |
| 澄清、缺失与不可回答 | 5 |
| 合规、安全与抗注入 | 6 |
| 实时数据与时效性 | 4 |
| 多轮与偏好记忆 | 4 |

## 建议评分

每条样本先执行硬门槛，再计算软分：

1. 硬门槛：不得出现 `forbidden_behaviors`，不得伪造事实或来源。
2. 固定事实题：事实正确 60%，证据正确 25%，任务完整 15%。
3. 综合题：结论受证据支持 45%，覆盖完整 30%，来源归因 15%，表达清晰 10%。
4. 动态题：时间戳和口径 25%，事实有可核验来源 40%，完整回答 25%，不确定性说明 10%。
5. 澄清/合规题：正确行为是主要判定，不因调用或未调用某个工具扣分。

路由判定分两级：

- `source_locked`：用户明确指定年报、本地库或某份报告，违反来源约束直接失败。
- `default`：必须落入正确能力域，但不绑定具体供应商。例如标准财务指标属于 `financial_metric`，将来从问财切换到另一家规范化财务库无需改测试集。

对于默认路由，工具调用只用于诊断；只要能力域、事实、证据均正确就通过。对于明确来源锁定，结果正确但来源错误仍不通过。

为保持 LangSmith 样本的 `outputs` 聚焦答案金标准，路由契约单独保存在 `datasets/fin-harness-core-v1-routing.jsonl`，通过相同的 `id` 一一关联。上传或运行评测时合并两者；不要把 `diagnostic_capabilities` 当成路由契约。

## 本地校验

```bash
python3 evals/validate_core_dataset.py
```

## 上传与运行

环境至少需要 Python 3.11、项目依赖，以及：

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=fin-harness-eval
```

幂等上传；已存在的 `case_id` 会跳过：

```bash
python3 -m evals.upload_dataset
```

默认运行 10 条校准样本：

```bash
python3 -m evals.run_core_eval
```

筛选和全量运行：

```bash
python3 -m evals.run_core_eval --case-id core-001 --case-id core-005
python3 -m evals.run_core_eval --bucket grounded_fact --limit 5
python3 -m evals.run_core_eval --all
```

确定性评估器不会从自然语言答案反向编造 `facts`、`citations` 或 `claims`；这些字段只来自 Harness session events 中的结构化工具结果。因此，如果某个 provider 没有输出结构化 facts/citations，分数会暴露该可观测性缺口。`required_behavior_semantic` 在未配置 LLM judge 时返回未评分，需在 LangSmith 中人工复核或后续接入经人工校准的 judge。

校验通过后再上传 LangSmith。建议把上传脚本做成 upsert，并以 `id` 作为稳定键；数据集版本变更时创建 `fin-harness-core-v2`，不要原地覆盖历史基线。
