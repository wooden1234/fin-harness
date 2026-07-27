# Agent V1

这里存放 V1 的根图实现。V1 采用固定 Supervisor 路由，入口为
`agent-v1.graph:get_graph`。

V1 与 V2 的隔离边界如下：

- 根图代码、编译缓存和 Studio graph 名称独立；
- checkpoint 通过 `graph_version` 和 `:graph:v2` thread 后缀隔离；
- `agents/graph.py` 仅作为旧调用方的兼容门面，不是 V1 实现；
- guardrails、工具、领域 Agent 等属于共享能力层，故意保持一致，确保两个版本
  的合规和工具治理规则不会分叉。

新增根图节点时，应只修改对应版本目录；跨版本能力变更应在共享层单独评审。
