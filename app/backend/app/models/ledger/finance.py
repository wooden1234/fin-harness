"""比赛账房业务表。放在 ledger 模式，不和 app.users 或年报事实表混用。"""

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY

from app.core.database import Base


class LedgerCompany(Base):
    __tablename__ = "companies"
    __table_args__ = {"schema": "ledger"}

    code = Column(String(16), primary_key=True)
    name = Column(String(128), nullable=False)
    short_name = Column(String(64), nullable=False)


class LedgerUser(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "ledger"}

    code = Column(String(16), primary_key=True)
    name = Column(String(64), nullable=False)


class LedgerPermission(Base):
    __tablename__ = "permissions"
    __table_args__ = {"schema": "ledger"}

    user_code = Column(String(16), primary_key=True)
    enterprise_code = Column(String(16), primary_key=True)
    can_read = Column(Integer, nullable=False)
    can_write = Column(Integer, nullable=False)


class LedgerInvoice(Base):
    __tablename__ = "invoices"
    __table_args__ = {"schema": "ledger"}

    enterprise_code = Column(String(16), primary_key=True)
    invoice_no = Column(String(32), primary_key=True)
    issued_on = Column(String(10), nullable=False)
    seller = Column(String(128), nullable=False)
    item_name = Column(String(128), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    purpose = Column(Text, nullable=False, default="")
    remark = Column(Text, nullable=False, default="")


class LedgerBankTxn(Base):
    __tablename__ = "bank_txns"
    __table_args__ = {"schema": "ledger"}

    enterprise_code = Column(String(16), primary_key=True)
    txn_no = Column(String(32), primary_key=True)
    txn_date = Column(String(10), nullable=False)
    direction = Column(String(8), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    counterparty = Column(String(128), nullable=False)
    remark = Column(Text, nullable=False, default="")


class LedgerRuleDoc(Base):
    """账房规则知识库。检索按这些列过滤，不靠正文相似度。"""

    __tablename__ = "rules"
    __table_args__ = {"schema": "ledger"}

    rule_id = Column(String(16), primary_key=True)
    version = Column(String(16), primary_key=True)
    name = Column(String(128), nullable=False)
    enterprises = Column(ARRAY(String(16)), nullable=False)
    effective_from = Column(String(10), nullable=False)
    status = Column(String(16), nullable=False)
    topic = Column(String(32), nullable=False)
    purpose_required = Column(String(32), nullable=True)
    purpose_threshold = Column(String(16), nullable=True)
    body = Column(Text, nullable=False)
    source_path = Column(Text, nullable=False)


class LedgerOpenTask(Base):
    __tablename__ = "open_tasks"
    __table_args__ = {"schema": "ledger"}

    app_user_id = Column(Integer, primary_key=True)
    enterprise_code = Column(String(16), primary_key=True)
    month = Column(String(7), nullable=False)
    missing_nos = Column(Text, nullable=False)
    updated_at = Column(String(40), nullable=False)


class LedgerSessionScope(Base):
    __tablename__ = "session_scope"
    __table_args__ = {"schema": "ledger"}

    session_id = Column(String(64), primary_key=True)
    user_code = Column(String(16), nullable=False)
    enterprise_code = Column(String(16), nullable=True)
