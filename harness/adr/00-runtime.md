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
- 用户可见正文只来自通过 `submit_answer` 的 `answer/published`。

## 相对 DSH 的产品分叉

DSH 无工具即可 `turn/end`。fin-agent 必须调用 `submit_answer`：闲聊 `mode=direct`；带数字必须 `grounded` 且引用本轮 tool/result 的 `evidence_id`。合规审查在发布前跑。

## 源权威

本地已入库年报与问财都是权威，差别是时效/覆盖。当前财务指标默认问财；点名已入库年报当时口径用 local-financial-facts；FAQ 不填数字；空结果不跳家族。

## 审批

`iwencai.*` 与财务事实工具 `requires_human_approval=True`。决策先落盘再执行。无人审批 fail-closed。迟到/重复 409 或幂等忽略。崩溃且已批准未出 result：合成 `cancelled_after_approval`，不再执行 handler。

## 长期偏好

每步 `assemble_system` 注入 `user_preferences` section（MemoryLoader + Redis Cache-Aside）。Loader 失败则空段，不阻断 loop。本轮「这次/本次」覆盖冲突 key。显式「请记住 / 改成 / 忘记」走 `memory_write` / `memory_delete`；暗示提取仍回答后出盒。不把读做成工具，不做情景 `memory_search`。

## Compaction

窗口 `COMPACTION_CONTEXT_WINDOW`（默认 65536）。阈值 `floor(window×0.8)`，尾部保留 `floor(window×0.16)`。先裁过大 tool/result，再事务式 summary。不拆 tool pair。overflow 最多重试 1 次。

## 组合

Python 包替换旧 `harness.runner` 对 LangGraph 的包装。产品 HTTP 只走 Agent loop。
