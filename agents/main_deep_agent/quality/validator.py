"""Main DeepAgent 的确定性证据门和安全答案渲染。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from agents.main_deep_agent.contracts import (
    MainAgentResponse,
    MainAgentStatement,
    MainAgentTable,
)
from agents.main_deep_agent.middleware.authorization import normalize_entity
from agents.main_deep_agent.quality.enrichments import (
    build_answer_charts,
    sanitize_follow_ups,
)
from agents.main_deep_agent.quality.renderer import (
    _clean_display_text,
    render_markdown_table,
)
from agents.main_deep_agent.quality.salvage import _render_salvage
from langchain_core.messages import HumanMessage
from agents.main_deep_agent.tools.evidence_adapter import detect_fact_conflicts
from agents.orchestrator.contracts import (
    AnswerStatement,
    Claim,
    ClaimEvidenceLink,
    ConstrainedAnswer,
    Evidence,
    QualityReport,
)
from agents.orchestrator.state import OrchestratorState

_ACTION_OUTPUT_MARKERS = (
    "买入", "卖出", "加仓", "减仓", "仓位", "买点", "卖点", "止损",
    "目标价", "低吸", "介入", "优先选择", "优先配置", "建议配置", "推荐购买",
)
_RAW_JSON_MARKERS = ('"configured"', '"results"', "{'answer':", '"answer":')
# 仅识别「已主动说明口径」的措辞；原币单位词（港元/美元）会出现在正常陈述里，不能用来跳过提示。
_CURRENCY_ANNOTATION_MARKERS = (
    "币种", "汇率", "原币", "不宜直接横比", "同币种", "多种币种",
)
_PERIOD_ANNOTATION_MARKERS = (
    "会计期间", "财年口径", "财年起止", "非同一", "报告期不同", "财季不一致",
    "不同会计期间",
)
def _used_fact_currencies(
    evidence_by_id: dict[str, Evidence],
    used_evidence_ids: list[str],
) -> list[str]:
    currencies: list[str] = []
    for evidence_id in used_evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        for fact in list(item.metadata.get("facts") or []):
            if not isinstance(fact, dict):
                continue
            currency = str(fact.get("currency") or "").strip().upper()
            if currency and currency not in currencies:
                currencies.append(currency)
    return currencies


def _used_fiscal_periods(
    evidence_by_id: dict[str, Evidence],
    used_evidence_ids: list[str],
) -> list[str]:
    periods: list[str] = []
    for evidence_id in used_evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        period = str(item.metadata.get("fiscal_period") or "").strip()
        if period and period not in periods:
            periods.append(period)
        for fact in list(item.metadata.get("facts") or []):
            if not isinstance(fact, dict):
                continue
            fact_period = str(fact.get("fiscal_period") or "").strip()
            if fact_period and fact_period not in periods:
                periods.append(fact_period)
    return periods


def _comparison_calibre_gaps(
    *,
    summary: str,
    evidence_by_id: dict[str, Evidence],
    used_evidence_ids: list[str],
) -> list[str]:
    """多币种/多会计期间时补充口径说明，不拒答、不删已通过内容。"""
    gaps: list[str] = []
    currencies = _used_fact_currencies(evidence_by_id, used_evidence_ids)
    periods = _used_fiscal_periods(evidence_by_id, used_evidence_ids)
    if len(currencies) > 1 and not any(
        marker in summary for marker in _CURRENCY_ANNOTATION_MARKERS
    ):
        joined = "/".join(currencies)
        gaps.append(
            f"引用数据涉及多种币种（{joined}）；上表按原币分列，同比/增速可直接对照，"
            "绝对金额不宜直接横比。若需同币种对比，请补充可核验汇率后再换算。"
        )
    if len(periods) > 1 and not any(
        marker in summary for marker in _PERIOD_ANNOTATION_MARKERS
    ):
        joined = "、".join(periods)
        gaps.append(
            f"各方披露期间不完全一致（{joined}），属不同会计期间口径；"
            "已给出对照供参考，不视为同一报告期的严格横比。"
        )
    return gaps


@dataclass(frozen=True, slots=True)
class _RejectedItem:
    """保存回修反馈所需的最小拒绝信息。"""

    location: str
    reason_code: str
    text: str
    declared_evidence_ids: tuple[str, ...] = ()
    valid_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _MainValidationResult:
    """聚合现有发布契约和内部回修判定，不形成新的公共协议。"""

    response: MainAgentResponse | None
    sanitized_response: MainAgentResponse | None
    claims: tuple[Claim, ...]
    claim_evidence_links: tuple[ClaimEvidenceLink, ...]
    constrained_answer: ConstrainedAnswer
    quality_report: QualityReport
    rejected_items: tuple[_RejectedItem, ...]
    used_evidence_ids: tuple[str, ...]
    conflicts: tuple[dict[str, Any], ...]
    independent_publisher_count: int
    open_research_incomplete: bool
    cross_period_mismatch: int
    accepted_evidence_content_count: int
    revisable: bool

    @property
    def trigger_codes(self) -> list[str]:
        return list(dict.fromkeys(item.reason_code for item in self.rejected_items))


def _claim_id(text: str, index: int) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"main-claim-{index}-{digest}"


def _normalize_response(response: MainAgentResponse | Any | None) -> MainAgentResponse | None:
    try:
        return (
            response
            if isinstance(response, MainAgentResponse)
            else MainAgentResponse.model_validate(response)
            if response is not None
            else None
        )
    except ValueError:
        return None


_METRIC_BRACKET_SUFFIX = re.compile(r"\[\d{6,8}\]\s*$")
_PERIOD_QUARTER_MARKERS: tuple[tuple[str, int], ...] = (
    ("一季度", 1), ("Q1", 1),
    ("中期", 2), ("半年度", 2), ("半年报", 2), ("H1", 2), ("二季度", 2),
    ("三季度", 3), ("前三季度", 3), ("Q3", 3),
    ("全年", 4), ("年度", 4), ("年报", 4), ("Q4", 4), ("四季度", 4),
)


def _normalize_metric_label(value: str) -> str:
    """去掉工具字段自带的 [YYYYMMDD] 取数口径后缀，只比较指标名本身。"""
    return _METRIC_BRACKET_SUFFIX.sub("", value).strip()


def _period_signature(period: str) -> tuple[str, int | None]:
    """把「2025年前三季度」「FY2025 Q4」等归一成 (年份, 季度粒度)；季度未知时为 None，不参与硬校验。"""
    year_match = re.search(r"(20\d{2})", period)
    year = year_match.group(1) if year_match else ""
    quarter: int | None = None
    for marker, value in _PERIOD_QUARTER_MARKERS:
        if marker in period:
            quarter = value
            break
    return year, quarter


def _statement_matches_financial_facts(
    statement: MainAgentStatement,
    cited: list[Evidence],
) -> bool:
    """财报数字必须能在 Evidence 的某条 fact 上找到对应实体/指标/期间/单位。

    比较采用归一化后的语义等价，而非原始字符串全等：模型写的是给用户看的
    自然标签（如「归母净利润同比增长率」「2025年前三季度」），工具内部字段
    则带取数口径后缀（如「…[20250930]」「FY2025」）；两者本就不应强制逐字符相同。
    """
    if not any(character.isdigit() for character in statement.text):
        return True
    if not (statement.entity and statement.metric and statement.period):
        return False
    target_entity = normalize_entity(statement.entity)
    stmt_metric = _normalize_metric_label(statement.metric)
    stmt_year, stmt_quarter = _period_signature(statement.period)
    for item in cited:
        for fact in list(item.metadata.get("facts") or []):
            if not isinstance(fact, dict):
                continue
            if normalize_entity(str(fact.get("entity") or "")) != target_entity:
                continue
            fact_metric = _normalize_metric_label(str(fact.get("metric") or ""))
            if fact_metric != stmt_metric and stmt_metric not in fact_metric and fact_metric not in stmt_metric:
                continue
            fact_year, fact_quarter = _period_signature(str(fact.get("fiscal_period") or ""))
            if stmt_year and fact_year and stmt_year != fact_year:
                continue
            if stmt_quarter is not None and fact_quarter is not None and stmt_quarter != fact_quarter:
                continue
            fact_unit = str(fact.get("unit") or "")
            if statement.unit and fact_unit and fact_unit != statement.unit:
                continue
            return True
    return False


def evaluate_main_response(
    response: MainAgentResponse | Any | None,
    evidence: list[Evidence],
    *,
    sensitive: bool,
) -> _MainValidationResult:
    """一次性完成逐项证据校验，并产出 gate 与回修共享的结果。"""
    normalized = _normalize_response(response)
    evidence_by_id = {item.evidence_id: item for item in evidence}

    def valid_calculation_lineage(item: Evidence) -> bool:
        """计算证据必须引用本轮真实存在的事实位置。"""
        if item.source_type != "calculation.run" or not item.metadata.get("formula"):
            return False
        input_ids = [
            str(value)
            for value in list(item.metadata.get("input_evidence_ids") or [])
        ]
        refs = list(item.metadata.get("operand_refs") or [])
        if not input_ids or len(refs) != 2:
            return False
        if any(evidence_id not in evidence_by_id for evidence_id in input_ids):
            return False
        for ref in refs:
            if not isinstance(ref, dict):
                return False
            evidence_id = str(ref.get("evidence_id") or "")
            source = evidence_by_id.get(evidence_id)
            if source is None:
                return False
            try:
                fact_index = int(ref.get("fact_index"))
            except (TypeError, ValueError):
                return False
            facts = list(source.metadata.get("facts") or [])
            if fact_index < 0 or fact_index >= len(facts):
                return False
        return True

    conflicts = detect_fact_conflicts(evidence)
    independent_publishers = {
        str(item.metadata.get("publisher_domain") or item.provider or "")
        for item in evidence
        if item.source_type != "calculation.run"
        and str(item.metadata.get("publisher_domain") or item.provider or "")
    }
    unresolved_conflict_ids = {
        evidence_id
        for conflict in conflicts
        if not conflict["resolved"]
        for evidence_id in conflict["evidence_ids"]
    }
    superseded_conflict_ids = {
        evidence_id
        for conflict in conflicts
        if conflict["resolution"] == "official_source_preferred"
        for evidence_id in conflict["evidence_ids"]
        if evidence_id != conflict["preferred_evidence_id"]
    }

    if normalized is None:
        return _MainValidationResult(
            response=None,
            sanitized_response=None,
            claims=(),
            claim_evidence_links=(),
            constrained_answer=ConstrainedAnswer(),
            quality_report=QualityReport(passed=False, reason="缺少有效结构化答案"),
            rejected_items=(),
            used_evidence_ids=(),
            conflicts=tuple(conflicts),
            independent_publisher_count=len(independent_publishers),
            open_research_incomplete=False,
            cross_period_mismatch=0,
            accepted_evidence_content_count=0,
            revisable=False,
        )

    if normalized.mode == "clarify":
        return _MainValidationResult(
            response=normalized,
            sanitized_response=normalized,
            claims=(),
            claim_evidence_links=(),
            constrained_answer=ConstrainedAnswer(),
            quality_report=QualityReport(passed=True, reason="需要用户补充关键信息"),
            rejected_items=(),
            used_evidence_ids=(),
            conflicts=tuple(conflicts),
            independent_publisher_count=len(independent_publishers),
            open_research_incomplete=False,
            cross_period_mismatch=0,
            accepted_evidence_content_count=0,
            revisable=False,
        )

    if normalized.mode == "direct":
        rejected = (
            _RejectedItem(
                location="direct_answer",
                reason_code="sensitive_direct_mode",
                text=normalized.direct_answer,
            ),
        ) if sensitive else ()
        direct_allowed = not evidence and not sensitive
        return _MainValidationResult(
            response=normalized,
            sanitized_response=normalized if direct_allowed else None,
            claims=(),
            claim_evidence_links=(),
            constrained_answer=ConstrainedAnswer(),
            quality_report=QualityReport(
                passed=direct_allowed,
                missing_evidence=[item.reason_code for item in rejected],
                unsupported_claim_ids=[item.reason_code for item in rejected],
                reason="无需外部事实的直接回答" if direct_allowed else "Direct 模式不满足发布条件",
            ),
            rejected_items=rejected,
            used_evidence_ids=(),
            conflicts=tuple(conflicts),
            independent_publisher_count=len(independent_publishers),
            open_research_incomplete=False,
            cross_period_mismatch=0,
            accepted_evidence_content_count=0,
            revisable=bool(rejected and evidence),
        )

    response_has_inference = any(
        statement.statement_type == "inference"
        for statement in normalized.statements
    )
    open_research_incomplete = response_has_inference and len(independent_publishers) < 2
    claims: list[Claim] = []
    links: list[ClaimEvidenceLink] = []
    answer_statements: list[AnswerStatement] = []
    accepted_statements: list[MainAgentStatement] = []
    accepted_tables: list[MainAgentTable] = []
    rejected_items: list[_RejectedItem] = []
    used_evidence_ids: list[str] = []
    accepted_evidence_content_count = 0
    verifiable_content_count = 0
    cross_period_mismatch = 0
    claim_index = 0

    def reject(
        *,
        location: str,
        reason_code: str,
        text: str,
        declared_ids: list[str],
        valid_ids: list[str],
    ) -> None:
        rejected_items.append(
            _RejectedItem(
                location=location,
                reason_code=reason_code,
                text=text,
                declared_evidence_ids=tuple(declared_ids),
                valid_evidence_ids=tuple(valid_ids),
            )
        )

    for statement_index, statement in enumerate(normalized.statements):
        location = f"statements[{statement_index}]"
        declared_ids = list(dict.fromkeys(statement.evidence_ids))
        valid_ids = [item for item in declared_ids if item in evidence_by_id]
        cited = [evidence_by_id[item] for item in valid_ids]
        if statement.statement_type != "caveat":
            verifiable_content_count += 1
        reason_code = ""
        if sensitive and any(marker in statement.text for marker in _ACTION_OUTPUT_MARKERS):
            reason_code = "investment_action_removed"
        elif statement.statement_type != "caveat" and not valid_ids:
            reason_code = "missing_evidence"
        elif statement.statement_type == "calculation" and not any(
            valid_calculation_lineage(item)
            for item in cited
        ):
            reason_code = "invalid_calculation_evidence"
        else:
            periods = {
                str(item.metadata.get("fiscal_period") or "")
                for item in cited
                if item.metadata.get("fiscal_period")
            }
            if statement.period and periods and (
                statement.period not in periods or periods != {statement.period}
            ):
                reason_code = "cross_period_mismatch"
                cross_period_mismatch += 1
            elif (
                any(item.metadata.get("facts") for item in cited)
                and any((statement.entity, statement.metric, statement.period))
                and not _statement_matches_financial_facts(statement, cited)
            ):
                reason_code = "financial_fact_mismatch"
            elif unresolved_conflict_ids.intersection(valid_ids):
                reason_code = "unresolved_conflict"
            elif superseded_conflict_ids.intersection(valid_ids):
                reason_code = "non_official_conflict_value"
        if reason_code:
            reject(
                location=location,
                reason_code=reason_code,
                text=statement.text,
                declared_ids=declared_ids,
                valid_ids=valid_ids,
            )
            continue

        accepted_statements.append(statement.model_copy(update={"evidence_ids": valid_ids}))
        if statement.statement_type != "caveat":
            accepted_evidence_content_count += 1
        claim_index += 1
        claim_id = _claim_id(statement.text, claim_index)
        claim_type = (
            "calculation" if statement.statement_type == "calculation"
            else "inference" if statement.statement_type == "inference"
            else "opinion" if statement.statement_type == "caveat"
            else "fact"
        )
        claims.append(
            Claim(
                claim_id=claim_id,
                task_id="main",
                text=statement.text,
                claim_type=claim_type,
            )
        )
        for evidence_id in valid_ids:
            links.append(
                ClaimEvidenceLink(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    relation="supports",
                    reason="MainAgentResponse 显式引用且通过一致性检查",
                )
            )
            if evidence_id not in used_evidence_ids:
                used_evidence_ids.append(evidence_id)
        answer_statements.append(
            AnswerStatement(
                text=statement.text,
                claim_ids=[claim_id],
                evidence_ids=valid_ids,
                statement_type=(
                    "inference" if statement.statement_type == "inference"
                    else "caveat" if statement.statement_type == "caveat"
                    else "fact"
                ),
                confidence=0.8 if valid_ids else 0.5,
            )
        )

    for table_index, table in enumerate(normalized.tables):
        accepted_rows = []
        for row_index, row in enumerate(table.rows):
            verifiable_content_count += 1
            location = f"tables[{table_index}].rows[{row_index}]"
            row_text = " | ".join(str(cell) for cell in row.cells)
            declared_ids = list(dict.fromkeys(row.evidence_ids))
            valid_ids = [item for item in declared_ids if item in evidence_by_id]
            reason_code = ""
            if sensitive and any(marker in row_text for marker in _ACTION_OUTPUT_MARKERS):
                reason_code = "investment_action_removed"
            elif not valid_ids or len(valid_ids) != len(declared_ids):
                reason_code = "table_row_missing_evidence"
            elif unresolved_conflict_ids.intersection(valid_ids):
                reason_code = "unresolved_conflict"
            elif superseded_conflict_ids.intersection(valid_ids):
                reason_code = "non_official_conflict_value"
            if reason_code:
                reject(
                    location=location,
                    reason_code=reason_code,
                    text=row_text,
                    declared_ids=declared_ids,
                    valid_ids=valid_ids,
                )
                continue

            accepted_rows.append(row.model_copy(update={"evidence_ids": valid_ids}))
            accepted_evidence_content_count += 1
            claim_index += 1
            claim_id = _claim_id(row_text, claim_index)
            claims.append(
                Claim(
                    claim_id=claim_id,
                    task_id="main",
                    text=row_text,
                    claim_type="fact",
                )
            )
            for evidence_id in valid_ids:
                links.append(
                    ClaimEvidenceLink(
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        relation="supports",
                        reason="MainAgentResponse 表格行显式引用且通过一致性检查",
                    )
                )
                if evidence_id not in used_evidence_ids:
                    used_evidence_ids.append(evidence_id)
            answer_statements.append(
                AnswerStatement(
                    text=row_text,
                    claim_ids=[claim_id],
                    evidence_ids=valid_ids,
                    statement_type="fact",
                    confidence=0.8,
                )
            )
        if accepted_rows:
            accepted_tables.append(table.model_copy(update={"rows": accepted_rows}))

    sanitized_response = (
        normalized.model_copy(
            update={
                "statements": accepted_statements,
                "tables": accepted_tables,
            }
        )
        if accepted_statements or accepted_tables
        else None
    )
    reason_codes = list(dict.fromkeys(item.reason_code for item in rejected_items))
    missing_evidence = [
        *reason_codes,
        *(["independent_publisher_missing"] if open_research_incomplete else []),
    ]
    coverage = (
        accepted_evidence_content_count / verifiable_content_count
        if verifiable_content_count
        else 0.0
    )
    return _MainValidationResult(
        response=normalized,
        sanitized_response=sanitized_response,
        claims=tuple(claims),
        claim_evidence_links=tuple(links),
        constrained_answer=ConstrainedAnswer(statements=answer_statements),
        quality_report=QualityReport(
            passed=accepted_evidence_content_count > 0,
            missing_evidence=list(dict.fromkeys(missing_evidence)),
            unsupported_claim_ids=reason_codes,
            claim_coverage=coverage,
            reason=(
                "证据检查通过"
                if accepted_evidence_content_count > 0
                else "没有可发布的证据型内容"
            ),
        ),
        rejected_items=tuple(rejected_items),
        used_evidence_ids=tuple(used_evidence_ids),
        conflicts=tuple(conflicts),
        independent_publisher_count=len(independent_publishers),
        open_research_incomplete=open_research_incomplete,
        cross_period_mismatch=cross_period_mismatch,
        accepted_evidence_content_count=accepted_evidence_content_count,
        revisable=bool(rejected_items),
    )


def _citation_markers(_evidence_ids: list[str], _used_ids: list[str]) -> str:
    """正文不再插入 [n] 角标；来源仍通过 citations 列表单独展示。"""
    return ""


def _is_brief_grounded(response: MainAgentResponse) -> bool:
    """单点事实：无表格、1–2 条非 caveat 陈述 → 散文结论。"""
    if response.tables:
        return False
    core = [
        item
        for item in response.statements
        if item.statement_type != "caveat"
    ]
    return 1 <= len(core) <= 2


def _render_brief_grounded(
    response: MainAgentResponse,
    used_ids: list[str],
) -> list[str]:
    """短事实题：加粗结论句，省略报告式 heading / bullet。"""
    core_lines: list[str] = []
    caveat_lines: list[str] = []
    for statement in response.statements:
        markers = _citation_markers(list(statement.evidence_ids), used_ids)
        line = f"{statement.text}{markers}"
        if statement.statement_type == "caveat":
            caveat_lines.append(line)
        else:
            core_lines.append(line)
    if not core_lines:
        return []
    if len(core_lines) == 1:
        body = f"**{core_lines[0]}**"
    else:
        body = f"**{core_lines[0]}**\n\n{core_lines[1]}"
    if caveat_lines:
        body = body + "\n\n" + "\n\n".join(caveat_lines)
    return [body]


def _render_accepted_response(
    evaluation: _MainValidationResult,
) -> list[str]:
    """只渲染校验结果中已接受的结构化内容。"""
    response = evaluation.sanitized_response
    if response is None or response.mode != "grounded":
        return []
    used_ids = list(evaluation.used_evidence_ids)
    if _is_brief_grounded(response):
        return _render_brief_grounded(response, used_ids)

    rendered_blocks: list[str] = []
    # 固定顺序：结论陈述 → 表格 → 口径/caveat，避免把「口径差异」渲染到表前。
    core_lines: list[str] = []
    caveat_lines: list[str] = []
    for statement in response.statements:
        markers = _citation_markers(list(statement.evidence_ids), used_ids)
        line = f"- {statement.text}{markers}"
        if statement.statement_type == "caveat":
            caveat_lines.append(line)
        else:
            core_lines.append(line)
    if core_lines:
        rendered_blocks.append("\n".join(core_lines))
    table_titles: list[str] = []
    for table in response.tables:
        rows = []
        for row in table.rows:
            markers = _citation_markers(list(row.evidence_ids), used_ids)
            rows.append((list(row.cells), markers))
        title = str(table.title or "").strip()
        if title.startswith("主表：") or title.startswith("主表:"):
            title = title.split("：", 1)[-1].split(":", 1)[-1].strip()
        if title:
            table_titles.append(title)
        rendered_table = render_markdown_table(
            title=title,
            columns=list(table.columns),
            rows=rows,
        )
        if rendered_table:
            rendered_blocks.append(rendered_table)
    if caveat_lines:
        rendered_blocks.append("\n".join(caveat_lines))
    if not rendered_blocks:
        return []
    heading = str(response.heading or "").strip()
    if heading.startswith("主表：") or heading.startswith("主表:"):
        heading = heading.split("：", 1)[-1].split(":", 1)[-1].strip()
    # 有表标题时不再叠 heading，避免「主要财务数据对比」+「A vs B」双标题。
    if response.tables and table_titles:
        if not heading or heading in table_titles or any(
            heading in title or title in heading for title in table_titles
        ):
            heading = ""
        elif heading in {"结论", "核心结论", "核心差异"}:
            pass
        else:
            # 表已有自然标题时，丢掉重复的说明性标题
            heading = ""
    body = "\n\n".join(rendered_blocks)
    return [f"### {heading}\n{body}" if heading else body]


async def main_evidence_quality_gate(state: OrchestratorState) -> dict[str, Any]:
    """按统一评价结果渲染答案，并在失败时安全降级。"""
    evidence = list(state.get("evidence") or [])
    sensitive = bool(state.get("investment_action_sensitive"))
    evaluation = evaluate_main_response(
        state.get("main_agent_response"),
        evidence,
        sensitive=sensitive,
    )
    response = evaluation.response
    conflicts = list(evaluation.conflicts)
    if response is not None and response.mode == "clarify":
        return {
            "summary": response.clarification,
            "execution_mode": "clarify",
            "execution_status": "clarify",
            "answer_follow_ups": [],
            "answer_charts": [],
            "quality_report": evaluation.quality_report,
            "main_quality_metrics": _quality_metrics(conflicts, "", False),
            "steps": ["main_deep_agent:evidence_quality_gate"],
        }
    if evaluation.sanitized_response is not None and response and response.mode == "direct":
        return {
            "summary": response.direct_answer,
            "execution_mode": "direct",
            "execution_status": "completed",
            "answer_follow_ups": [],
            "answer_charts": [],
            "quality_report": evaluation.quality_report,
            "main_quality_metrics": _quality_metrics(conflicts, response.direct_answer, False),
            "steps": ["main_deep_agent:evidence_quality_gate"],
        }

    rendered_sections = (
        _render_accepted_response(evaluation)
        if evaluation.accepted_evidence_content_count > 0
        else []
    )
    output_gaps: list[str] = []
    completed_with_gaps = False
    evidence_by_id = {item.evidence_id: item for item in evidence}
    used_evidence_ids = list(evaluation.used_evidence_ids)
    claims = list(evaluation.claims)
    links = list(evaluation.claim_evidence_links)
    answer_statements = list(evaluation.constrained_answer.statements)

    if rendered_sections:
        gaps: list[str] = []
        if response and response.gaps:
            gaps.append("部分补充资料未完成核验，不影响上述已引用事实。")
        if evaluation.rejected_items:
            gaps.append("部分候选陈述因证据、期间或冲突校验未通过，已从答案中移除。")
        if any(not item["resolved"] for item in conflicts):
            gaps.append("存在无法统一口径的数值冲突，相关陈述已从答案中移除。")
        if evaluation.open_research_incomplete:
            gaps.append("开放判断还需要第二个独立来源，当前来源偏少。")
        draft_summary = "\n\n".join(rendered_sections)
        gaps.extend(
            _comparison_calibre_gaps(
                summary=draft_summary,
                evidence_by_id=evidence_by_id,
                used_evidence_ids=used_evidence_ids,
            )
        )
        output_gaps = list(dict.fromkeys(gaps))
        if output_gaps:
            completed_with_gaps = True
            rendered_sections.append(
                "### 资料说明\n" + "\n".join(f"- {item}" for item in output_gaps)
            )
        summary = "\n\n".join(rendered_sections)
        execution_mode = (
            "deep_research"
            if evaluation.independent_publisher_count >= 2
            else "tool_assisted"
        )
        execution_status = "completed_with_gaps" if completed_with_gaps else "completed"
        quality_passed = True
    else:
        summary, gaps = _render_salvage(state)
        if evaluation.open_research_incomplete:
            publisher_gap = "开放判断还需要第二个独立来源，当前来源偏少。"
            if publisher_gap not in gaps:
                gaps.append(publisher_gap)
                summary += f"\n- {publisher_gap}"
        output_gaps = list(dict.fromkeys(gaps))
        displayable_ids = [
            item.evidence_id for item in evidence if _clean_display_text(item)
        ][:10]
        completed_with_gaps = bool(displayable_ids and not sensitive)
        execution_mode = (
            "tool_assisted" if completed_with_gaps
            else "partial" if evidence
            else "clarify"
        )
        execution_status = (
            "completed_with_gaps" if completed_with_gaps
            else "partial" if evidence
            else "failed"
        )
        quality_passed = completed_with_gaps
        used_evidence_ids = displayable_ids
        claims = []
        links = []
        answer_statements = (
            [
                AnswerStatement(
                    text="已按实际覆盖保留可核验资料，并明确未完成项。",
                    evidence_ids=used_evidence_ids,
                    statement_type="caveat",
                    confidence=0.6,
                )
            ]
            if used_evidence_ids
            else []
        )

    citation_number = {
        item: index for index, item in enumerate(used_evidence_ids, start=1)
    }
    citations = []
    for evidence_id in used_evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        citations.append(
            {
                "source": item.title or item.provider,
                "title": item.title or item.provider,
                "snippet": _clean_display_text(item) or "结构化工具结果（内部字段已隐藏）",
                "source_type": item.source_type,
                "sub_task_id": "main",
                "evidence_id": item.evidence_id,
                "citation_number": citation_number[evidence_id],
                **({"url": item.url} if item.url else {}),
                **({"published_at": item.published_at} if item.published_at else {}),
            }
        )

    multi_currency = len(_used_fact_currencies(evidence_by_id, used_evidence_ids)) > 1
    multi_fiscal_period = len(_used_fiscal_periods(evidence_by_id, used_evidence_ids)) > 1
    metrics = _quality_metrics(
        conflicts,
        summary,
        completed_with_gaps,
        cross_period_mismatch=evaluation.cross_period_mismatch,
        multi_currency_comparison=multi_currency,
        multi_fiscal_period_comparison=multi_fiscal_period,
    )
    missing_evidence = list(evaluation.quality_report.missing_evidence)
    query = ""
    for message in reversed(list(state.get("messages") or [])):
        if isinstance(message, HumanMessage):
            query = str(message.content or "").strip()
            break
    used_evidence = [
        item for item in evidence if item.evidence_id in set(used_evidence_ids)
    ] or list(evidence)
    model_follow_ups = (
        list(getattr(evaluation.sanitized_response, "follow_ups", None) or [])
        if evaluation.sanitized_response is not None
        else list(getattr(response, "follow_ups", None) or [])
        if response is not None
        else []
    )
    follow_ups = sanitize_follow_ups(
        model_follow_ups,
        query=query,
        investment_action_sensitive=sensitive,
    )
    answer_charts = build_answer_charts(
        evidence=used_evidence,
        used_evidence_ids=used_evidence_ids,
    )
    return {
        "summary": summary,
        "claims": claims,
        "claim_evidence_links": links,
        "constrained_answer": ConstrainedAnswer(
            statements=answer_statements,
            caveats=output_gaps,
        ),
        "citations": citations,
        "answer_follow_ups": follow_ups,
        "answer_charts": answer_charts,
        "quality_report": evaluation.quality_report.model_copy(
            update={
                "passed": quality_passed,
                "missing_evidence": missing_evidence,
                "reason": "证据检查通过" if quality_passed else "已按 journal 安全降级",
            }
        ),
        "main_quality_metrics": metrics,
        "execution_mode": execution_mode,
        "execution_status": execution_status,
        "steps": ["main_deep_agent:evidence_quality_gate"],
    }


def _quality_metrics(
    conflicts: list[dict[str, Any]],
    summary: str,
    completed_with_gaps: bool,
    *,
    cross_period_mismatch: int = 0,
    multi_currency_comparison: bool = False,
    multi_fiscal_period_comparison: bool = False,
) -> dict[str, Any]:
    return {
        "cross_period_mismatch": cross_period_mismatch,
        "multi_currency_comparison": multi_currency_comparison,
        "multi_fiscal_period_comparison": multi_fiscal_period_comparison,
        "raw_json_leak": any(marker in summary for marker in _RAW_JSON_MARKERS),
        "conflict_detected": len(conflicts),
        "conflict_resolved": sum(bool(item["resolved"]) for item in conflicts),
        "completed_with_gaps": completed_with_gaps,
    }


__all__ = ["evaluate_main_response", "main_evidence_quality_gate"]
