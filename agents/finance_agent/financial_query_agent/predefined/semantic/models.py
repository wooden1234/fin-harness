"""财务语义层数据模型。

语义层位于 extraction（槽位抽取）与 sql_builder（SQL 生成）之间，负责：
1. 指标标准化：用户说法 → canonical 统一语义 → 公司级 DB 字段
2. 覆盖判断：该公司/年份/粒度下是否有数据，以及取数策略

数据流转：
  FinancialQueryIntent（上游意图）
    → CanonicalMetricMatch（指标映射结果）
    → CoverageResolution（覆盖与口径结论）
    → ResolvedMetricBinding（下游 SQL 绑定参数）
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.finance_agent.financial_query_agent.predefined.semantic.reason_codes import (
    CoverageReasonCode,
)

# ---------------------------------------------------------------------------
# 枚举类型
# ---------------------------------------------------------------------------

# 覆盖解析的整体结论：当前查询能否执行、是否需要用户澄清
CoverageStatus = Literal[
    "ok",           # 所有目标公司均可查，可直接生成 SQL
    "partial",      # 部分公司可查（常见于 compare），可降级返回答案
    "clarify",      # 存在多种口径/粒度，需追问用户后再查
    "unavailable",  # 完全查不到，终止白名单路径
]

# 单家公司的取数策略：从 DB 里用什么方式拿到指标值
CoverageStrategy = Literal[
    "annual_direct",            # 直接取年报（period_type=annual）数据
    "sum_quarters",             # 已废弃：predefined 不再做四季汇总
    "quarter_only",             # 仅有季度、无年报 → 应回退 text_to_sql
    "latest_annual",            # 取最近一期年报
    "latest_available",         # 已废弃：predefined 不再接受非年报最新值
    "partial_compare",          # 对比场景下部分公司缺数据
    "clarify_for_granularity",  # 年报/季报等多种粒度均可用，需用户确认
    "unavailable",              # 该公司该指标无可用年报数据
]

# 指标名映射来源：全局别名表 vs 公司级覆盖规则
MatchType = Literal[
    "global_alias",     # 命中 registry_seed.GLOBAL_ALIASES，如「营收」→ REVENUE
    "company_override", # 命中 registry_seed.COMPANY_OVERRIDES，如腾讯 REVENUE →「收入」
]

# 查询操作类型，与 extraction / tool_selection 的 operation 字段一致
QueryType = Literal[
    "lookup",        # 查指定年份的单值
    "latest",        # 查最新一期
    "compare",       # 多公司横向对比（同年）
    "compare_year",  # 单公司跨年对比
    "trend",         # 多年历史趋势
]

# 答案生成策略：formatter 据此决定是否附加口径说明
AnswerPolicy = Literal[
    "direct",                              # 直接展示查数结果，无需额外说明
    "compare_with_mixed_source_metrics",   # 对比公司使用了不同口径字段，需提示
    "partial_compare",                     # 部分公司缺数据，对比结果不完整
    "sum_quarters_disclosure",             # 已废弃：predefined 不再汇总季度
    "trend_with_gaps",                     # 趋势序列存在年份缺口
    "clarify_for_granularity",             # 需用户确认查询粒度后再答
    "unavailable",                         # 无法生成有效答案
]


# ---------------------------------------------------------------------------
# 注册表相关模型（静态种子 → 运行时映射）
# ---------------------------------------------------------------------------

class CanonicalMetricDefinition(BaseModel):
    """统一财务语义定义，对应 registry_seed.CANONICAL_METRICS 中的一条。

    例：code="REVENUE", name="营业收入"
    无论用户说「营收」「收入」还是「营业额」，最终都归一到此 code。
    """

    code: str           # 内部标准编码，如 REVENUE、NET_INCOME_ATTR_PARENT
    name: str           # 中文标准名，如「营业收入」「归母净利润」
    description: str = ""  # 口径说明，如「合并报表营业收入」


class CompanyMetricOverride(BaseModel):
    """公司级指标名覆盖规则，对应 registry_seed.COMPANY_OVERRIDES 中的一条。

    同一 canonical 语义在不同公司报表里字段名不同，由此指定。
    例：Tencent 的 REVENUE 在 DB 里实际叫「收入」而非「营业收入」。
    """

    company_key: str            # 语义层公司 key，如 Tencent、CATL
    canonical_metric_code: str  # 统一语义编码，如 REVENUE
    metric_name: str            # 该公司 DB 中的实际字段名，如「收入」


# ---------------------------------------------------------------------------
# 指标映射结果（canonical_metric_registry 产出）
# ---------------------------------------------------------------------------

class CompanyMetricMatch(BaseModel):
    """单个公司下，一个 canonical 指标映射到 DB 的具体结果。"""

    company_key: str              # 语义层公司 key，如 Tencent
    company_id: int | None = None # financial_companies.id；未解析到则为 None
    metric_id: int | None = None  # financial_metrics.id；未在 DB 找到则为 None
    metric_name: str              # DB 中匹配到的字段名，如「收入」「营业收入(千元)」
    match_type: MatchType         # 映射来源：全局别名 or 公司覆盖
    confidence: float = Field(ge=0.0, le=1.0)  # 匹配置信度，0.95~0.98


class CanonicalMetricMatch(BaseModel):
    """一个用户请求的指标，标准化后的完整映射结果。

    例：用户说「营收」→ canonical REVENUE → 各公司的 CompanyMetricMatch 列表。
    """

    canonical_metric_code: str   # 统一编码，如 REVENUE；无法识别时为空串
    canonical_metric_name: str   # 统一中文名，如「营业收入」
    requested_metric: str        # 用户原始说法，如「营收」
    company_metric_matches: list[CompanyMetricMatch] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 覆盖解析（coverage_resolver 输入 / 输出）
# ---------------------------------------------------------------------------

class CoverageRequest(BaseModel):
    """覆盖解析的输入：在指标已映射后，判断数据是否可查。"""

    canonical_matches: list[CanonicalMetricMatch]  # 上游指标映射结果
    companies: list[str]                           # 目标公司列表
    years: list[int] = Field(default_factory=list) # 目标年份；空表示由策略自动选取
    query_type: QueryType = "lookup"               # 查询操作类型
    template_id: str = ""                          # 白名单模板 ID，如 exact_metric_lookup


class CompanyCoverage(BaseModel):
    """单家公司的数据覆盖情况与选定取数策略。"""

    company_key: str              # 语义层公司 key
    company_id: int               # DB 公司 ID
    metric_id: int                # DB 指标 ID
    canonical_metric_code: str    # 统一语义编码
    metric_name: str              # DB 实际字段名
    available_period_types: list[str] = Field(
        default_factory=list,
        description="DB 中已有的期间类型，如 annual、quarter、period_end",
    )
    available_years: list[int] = Field(
        default_factory=list,
        description="有数据的年份列表",
    )
    selected_strategy: CoverageStrategy = "unavailable"  # 最终选用的取数策略
    selected_year: int | None = None                       # 最终选用的年份


class CoverageResolution(BaseModel):
    """覆盖解析的整体结论，供 resolver / formatter 消费。

    status=clarify 时 predefined_workflow 会直接返回 clarify_reason 追问用户。
    status=unavailable 时终止白名单路径。
    """

    status: CoverageStatus = "unavailable"
    canonical_metric_code: str = ""
    company_coverages: list[CompanyCoverage] = Field(default_factory=list)
    answer_policy: AnswerPolicy = "unavailable"  # 答案格式化策略
    reason_code: CoverageReasonCode | None = None  # 机器可读结论码，供路由/监控
    clarify_reason: str = ""       # 展示给用户的追问或 partial 说明文案
    unavailable_reason: str = ""   # status=unavailable 时的原因说明


# ---------------------------------------------------------------------------
# SQL 绑定参数（resolver 产出，sql_builder / execution 消费）
# ---------------------------------------------------------------------------

class ResolvedMetricBinding(BaseModel):
    """一条可执行的「公司 + 指标 + 取数策略」绑定，用于生成 SQL。

    由 resolver.build_metric_bindings() 从 CanonicalMetricMatch + CoverageResolution 组装。
    """

    company_id: int                          # DB 公司 ID
    company_key: str = ""                  # 语义层公司 key
    metric_id: int                         # DB 指标 ID
    canonical_metric_code: str             # 统一语义编码
    metric_name: str = ""                  # DB 实际字段名
    selected_strategy: CoverageStrategy = "annual_direct"  # 取数策略
    selected_year: int | None = None       # 目标年份


__all__ = [
    "AnswerPolicy",
    "CanonicalMetricDefinition",
    "CanonicalMetricMatch",
    "CompanyCoverage",
    "CompanyMetricMatch",
    "CompanyMetricOverride",
    "CoverageRequest",
    "CoverageReasonCode",
    "CoverageResolution",
    "CoverageStatus",
    "CoverageStrategy",
    "MatchType",
    "QueryType",
    "ResolvedMetricBinding",
]
