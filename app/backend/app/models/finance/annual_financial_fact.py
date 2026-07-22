from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Float,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class FinancialCompany(Base):
    """Normalized company dimension for annual-report facts."""

    __tablename__ = "financial_companies"
    __table_args__ = {"schema": "fin_core"}

    id = Column(Integer, primary_key=True, index=True)
    # 公司规范化唯一标识，用于合并别名和跨文档归一。
    company_key = Column(String(255), nullable=False, unique=True, index=True)
    # 公司展示名称。
    name = Column(String(255), nullable=False, index=True)
    # 股票代码；部分文档源可能缺失，因此允许为空。
    ticker = Column(String(32), nullable=True, index=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    documents = relationship("AnnualReportDocument", back_populates="company")


class AnnualReportDocument(Base):
    """Annual report document metadata split out of fact rows."""

    __tablename__ = "annual_report_documents"
    __table_args__ = {"schema": "fin_core"}

    id = Column(Integer, primary_key=True, index=True)
    # 外部文档唯一标识，用于和原始解析产物稳定对齐。
    doc_id = Column(String(80), nullable=False, unique=True, index=True)
    # 所属公司主键；未完成归属时允许为空。
    company_id = Column(
        Integer,
        ForeignKey("fin_core.financial_companies.id"),
        nullable=True,
        index=True,
    )
    # 文档标题，通常是年报名称。
    title = Column(String(255), nullable=False)
    # 文档对应财年。
    fiscal_year = Column(Integer, nullable=True, index=True)
    # 原始文件名或来源标识，便于追溯。
    source = Column(String(255), nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    company = relationship("FinancialCompany", back_populates="documents")
    tables = relationship("AnnualFinancialTable", back_populates="document")


class AnnualFinancialTable(Base):
    """Table/chunk context shared by many extracted financial facts."""

    __tablename__ = "annual_financial_tables"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_annual_financial_table_document_chunk",
        ),
        {"schema": "fin_core"},
    )

    id = Column(Integer, primary_key=True, index=True)
    # 所属年报文档主键。
    document_id = Column(
        Integer,
        ForeignKey("fin_core.annual_report_documents.id"),
        nullable=False,
        index=True,
    )
    # 文档内表格分块序号，用于保证定位稳定。
    chunk_index = Column(Integer, nullable=False)
    # 表格所在页码。
    page_num = Column(Integer, nullable=True)
    # 表格所在章节或标题。
    section = Column(String(255), nullable=True)
    # 表格类别，如利润表、资产负债表、现金流量表。
    table_kind = Column(String(64), nullable=False, index=True)
    # 原始表格文本，便于回溯抽取上下文。
    raw_table_text = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    document = relationship("AnnualReportDocument", back_populates="tables")
    facts = relationship("AnnualFinancialFact", back_populates="table")


class FinancialMetric(Base):
    """Source financial metric dictionary extracted from reports."""

    __tablename__ = "financial_metrics"
    __table_args__ = {"schema": "fin_core"}

    id = Column(Integer, primary_key=True, index=True)
    # PDF 抽取出的原始指标名；历史字段名保留 canonical_name，语义上不再等同于统一口径。
    canonical_name = Column(String(255), nullable=False, unique=True, index=True)
    # 原始抽取阶段提供的别名；统一口径别名应维护在 canonical_metric_aliases。
    aliases = Column(String(512), nullable=True)
    # 原始表格或章节类型，用于缩小 source metric 匹配范围。
    statement_type = Column(String(128), nullable=True, index=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    facts = relationship("AnnualFinancialFact", back_populates="metric")


class CanonicalMetric(Base):
    """Enterprise canonical metric registry used by agents."""

    __tablename__ = "canonical_metrics"
    __table_args__ = {"schema": "fin_core"}

    # 稳定业务编码，如 REVENUE、NET_INCOME_ATTR_PARENT。
    code = Column(String(64), primary_key=True)
    # 面向用户展示的统一指标名。
    name = Column(String(255), nullable=False, index=True)
    # 所属报表类型，用于限制可比口径。
    statement_type = Column(String(64), nullable=True, index=True)
    # 指标值类型，如 amount、ratio、count。
    value_type = Column(String(32), nullable=False, default="amount")
    # 默认单位；实际查询仍以 fact.unit/source 披露为准。
    default_unit = Column(String(64), nullable=True)
    # 指标口径说明。
    description = Column(Text, nullable=True)
    # 是否在 agent 查询中启用。
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    aliases = relationship("CanonicalMetricAlias", back_populates="metric")
    company_mappings = relationship("CompanyMetricMapping", back_populates="canonical_metric")
    facts = relationship("AnnualFinancialFact", back_populates="canonical_metric")


class CanonicalMetricAlias(Base):
    """User-facing aliases for canonical metric resolution."""

    __tablename__ = "canonical_metric_aliases"
    __table_args__ = (
        UniqueConstraint("alias", name="uq_canonical_metric_alias"),
        {"schema": "fin_core"},
    )

    id = Column(Integer, primary_key=True, index=True)
    canonical_code = Column(
        String(64),
        ForeignKey("fin_core.canonical_metrics.code"),
        nullable=False,
        index=True,
    )
    # 用户可能输入的指标词，如“营收”“收入”“归母净利”。
    alias = Column(String(255), nullable=False, index=True)
    # 归一化后的别名，用于大小写/空格不敏感匹配。
    normalized_alias = Column(String(255), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="seed")
    priority = Column(Integer, nullable=False, default=100)
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    metric = relationship("CanonicalMetric", back_populates="aliases")


class CompanyMetricMapping(Base):
    """Company-specific mapping from canonical metric to source metric."""

    __tablename__ = "company_metric_mappings"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "canonical_code",
            "source_metric_id",
            name="uq_company_metric_mapping_source",
        ),
        {"schema": "fin_core"},
    )

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(
        Integer,
        ForeignKey("fin_core.financial_companies.id"),
        nullable=False,
        index=True,
    )
    canonical_code = Column(
        String(64),
        ForeignKey("fin_core.canonical_metrics.code"),
        nullable=False,
        index=True,
    )
    source_metric_id = Column(
        Integer,
        ForeignKey("fin_core.financial_metrics.id"),
        nullable=False,
        index=True,
    )
    # 冗余保存源指标名，便于审计和人工审核界面展示。
    source_metric_name = Column(String(255), nullable=False)
    statement_type = Column(String(128), nullable=True, index=True)
    valid_from_year = Column(Integer, nullable=True)
    valid_to_year = Column(Integer, nullable=True)
    priority = Column(Integer, nullable=False, default=100)
    confidence = Column(Float, nullable=False, default=0.9)
    mapping_source = Column(String(32), nullable=False, default="seed")
    review_status = Column(String(32), nullable=False, default="approved", index=True)
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    company = relationship("FinancialCompany")
    canonical_metric = relationship("CanonicalMetric", back_populates="company_mappings")
    source_metric = relationship("FinancialMetric")


class RawTableCell(Base):
    """Cell-level table extraction evidence for financial facts."""

    __tablename__ = "raw_table_cells"
    __table_args__ = (
        UniqueConstraint(
            "table_id",
            "row_index",
            "col_index",
            name="uq_raw_table_cell_position",
        ),
        {"schema": "fin_core"},
    )

    id = Column(Integer, primary_key=True, index=True)
    table_id = Column(
        Integer,
        ForeignKey("fin_core.annual_financial_tables.id"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        Integer,
        ForeignKey("fin_core.annual_report_documents.id"),
        nullable=False,
        index=True,
    )
    page_num = Column(Integer, nullable=True)
    row_index = Column(Integer, nullable=False)
    col_index = Column(Integer, nullable=False)
    row_header = Column(String(512), nullable=True)
    col_header = Column(String(512), nullable=True)
    cell_text = Column(Text, nullable=True)
    normalized_value = Column(Numeric(24, 6), nullable=True)
    unit = Column(String(64), nullable=True)
    currency = Column(String(32), nullable=True)
    # JSON 字符串，保留页内坐标；当前抽取链路没有坐标时允许为空。
    bbox_json = Column(Text, nullable=True)
    extractor = Column(String(64), nullable=False, default="pdf_pipeline")
    extract_version = Column(String(64), nullable=False, default="v1")
    confidence = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AnnualFinancialFact(Base):
    """Narrow fact table: one metric value for one period in one source table."""

    __tablename__ = "annual_financial_facts"
    __table_args__ = (
        UniqueConstraint(
            "table_id",
            "row_index",
            "metric_id",
            "period_label",
            name="uq_annual_financial_fact_source_metric",
        ),
        {"schema": "fin_core"},
    )

    id = Column(Integer, primary_key=True, index=True)
    # 来源财务表主键。
    table_id = Column(
        Integer,
        ForeignKey("fin_core.annual_financial_tables.id"),
        nullable=False,
        index=True,
    )
    # 对应指标主键。
    metric_id = Column(
        Integer,
        ForeignKey("fin_core.financial_metrics.id"),
        nullable=False,
        index=True,
    )
    # 统一指标编码；只有经过语义映射的数据才会填充，agent 查询应优先使用该字段。
    canonical_code = Column(
        String(64),
        ForeignKey("fin_core.canonical_metrics.code"),
        nullable=True,
        index=True,
    )
    # 单元格级来源证据；旧数据可为空，后续抽取链路应逐步补齐。
    source_cell_id = Column(
        Integer,
        ForeignKey("fin_core.raw_table_cells.id"),
        nullable=True,
        index=True,
    )

    # 指标所在原始表行序号，用于区分同表内多行事实。
    row_index = Column(Integer, nullable=False, default=0)
    # 原始期间标签，如“2024年”“本期”“上年同期”。
    period_label = Column(String(128), nullable=True, index=True)
    # 标准化后的年份，便于结构化筛选。
    period_year = Column(Integer, nullable=True, index=True)
    # 期间类型，如 annual、quarterly；主要用于清洗和过滤。
    period_type = Column(String(64), nullable=True)
    # 标准化后的数值结果，便于排序和计算。
    value = Column(Numeric(24, 6), nullable=True)
    # 原始文本值，保留展示口径和人工核对依据。
    raw_value = Column(String(128), nullable=True)
    # 数值单位，如元、万元、千元。
    unit = Column(String(64), nullable=True)
    # 币种信息，如人民币、美元。
    currency = Column(String(32), nullable=True)
    # 原始整行文本，便于回溯抽取上下文。
    raw_row = Column(Text, nullable=True)
    # 抽取/映射置信度；用于低置信度事实进入人工审核队列。
    confidence = Column(Float, nullable=False, default=0.0)
    # 质量校验状态：pending、passed、failed、reviewed。
    quality_status = Column(String(32), nullable=False, default="pending", index=True)
    # 人工审核状态：unreviewed、approved、rejected。
    review_status = Column(String(32), nullable=False, default="unreviewed", index=True)
    # 校验失败或告警原因，JSON 字符串。
    validation_errors = Column(Text, nullable=True)
    # 抽取规则/模型版本，保证清洗结果可回放。
    extract_version = Column(String(64), nullable=False, default="v1")
    # 发布开关；agent 精确查询只应使用已发布事实。
    is_published = Column(Boolean, nullable=False, default=False, index=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    table = relationship("AnnualFinancialTable", back_populates="facts")
    metric = relationship("FinancialMetric", back_populates="facts")
    canonical_metric = relationship("CanonicalMetric", back_populates="facts")
    source_cell = relationship("RawTableCell")
