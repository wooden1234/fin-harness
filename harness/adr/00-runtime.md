# ADR：fin-agent Agent-in-the-Loop 运行时

状态：已接受  
范围：本目录 `harness/` 与产品 `POST /api/agent/*`。改其中任一条先改本文件。

## 对标

DeepSeek Harness 的 session log、turn/step、tool pipeline、`derive_messages`。  
**不**移植 Cordis、web client、Typert；**不**用 DSH Python SDK（那是包 Node）。

## 不变量

- Model-visible ⟺ logged。`derive_messages` 是模型历史的唯一来源。
- 每个 `tool/call` 恰好一条 `tool/result`（含取消、拒绝合成）。
- 同一 session 同时一个 turn owner；抢租约失败 HTTP 409。
- 候选 `assistant/chunk` 不得成为用户可见输出。
- 用户可见正文只来自无工具调用的助手正文写入的 `answer/published`。工具结果上的 `evidence_id` 保留在 session log，不作为发布闸门。

## 相对 DSH 的产品分叉

DSH 无工具即可 `turn/end`。fin-agent 同样：模型不再调用 `submit_answer`；无 `tool_calls` 且有助手正文即发布。合规审查仍在发布前跑。`evidence_id` 由工具 `stamp_evidence` 写入 `tool/result`，供追踪与评测，不校验回答是否引用。

## finalign

本地 `finalign-awq` 不是 loop 主模型，也不是规划器。DeepSeek 负责选工具；收齐多路 `tool/result` 后，若 finalign 可达则调用 `finalign.analyze` 成稿。vLLM 不可达时**不把该工具交给模型、不回退 DeepSeek**，规划模型根据本轮工具结果直接用正文作答，复用 loop 的 KV 前缀。

## 源权威

本地已入库年报与问财都是权威，差别是时效/覆盖。当前财务指标默认问财；点名已入库年报当时口径用 local-financial-facts；FAQ 不填数字；空结果不跳家族。

## 审批

`iwencai.*` 与财务事实工具 `requires_human_approval=True`。决策先落盘再执行。无人审批 fail-closed。迟到/重复 409 或幂等忽略。崩溃且已批准未出 result：合成 `cancelled_after_approval`，不再执行 handler。

## 长期偏好

每步 `assemble_system` 在稳定前缀（身份、合规、工具纪律、skill 目录）之后注入偏好：长期偏好 `order=50`，本轮「这次/本次」`order=60`。两者都不插入工具纪律之前，以免打冷 KV 前缀。Loader（MemoryLoader + Redis Cache-Aside）失败则空段，不阻断 loop。显式「请记住 / 改成 / 忘记」走 `memory_write` / `memory_delete`；暗示提取仍回答后出盒。不把读做成工具，不做情景 `memory_search`。

## 工具错误分类

失败 payload 一律带 `error` / `error_class` / `model_guidance`。分类与处理：

- `transient`：超时网络，允许同工具再试一次
- `invalid_input`：改参数或向用户澄清，禁止原样重试
- `empty`：换更匹配来源或如实缺口
- `unavailable`：数据源未配置/挂掉，不要再打同一工具
- `policy`：未知工具或重试耗尽；本轮无成功数据则直接回复用户
- `compensate`：finalign 不可用；不注入、不另开 LLM，主路径根据已有 tool/result 直接作答
- `control`：取消、等待审批

## Compaction

窗口 `COMPACTION_CONTEXT_WINDOW`（默认 65536）。阈值 `floor(window×0.8)`，尾部保留 `floor(window×0.16)`。先裁过大 tool/result，再事务式 summary。不拆 tool pair。overflow 最多重试 1 次。

## 组合

Python 包替换旧 `harness.runner` 对 LangGraph 的包装。产品 HTTP 只走 Agent loop。
