BEGIN;

-- Run this only when the old wide annual_financial_facts table already exists.
ALTER TABLE annual_financial_facts RENAME TO annual_financial_facts_legacy;
ALTER SEQUENCE IF EXISTS annual_financial_facts_id_seq
    RENAME TO annual_financial_facts_legacy_id_seq;
ALTER TABLE annual_financial_facts_legacy
    RENAME CONSTRAINT uq_annual_financial_fact_source_metric
    TO uq_annual_financial_fact_source_metric_legacy;

CREATE TABLE IF NOT EXISTS financial_companies (
    id SERIAL PRIMARY KEY,
    company_key VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(32),
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_financial_companies_company_key
    ON financial_companies (company_key);
CREATE INDEX IF NOT EXISTS ix_financial_companies_name
    ON financial_companies (name);
CREATE INDEX IF NOT EXISTS ix_financial_companies_ticker
    ON financial_companies (ticker);

CREATE TABLE IF NOT EXISTS annual_report_documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(80) NOT NULL UNIQUE,
    company_id INTEGER REFERENCES financial_companies(id),
    title VARCHAR(255) NOT NULL,
    fiscal_year INTEGER,
    source VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_annual_report_documents_doc_id
    ON annual_report_documents (doc_id);
CREATE INDEX IF NOT EXISTS ix_annual_report_documents_company_id
    ON annual_report_documents (company_id);
CREATE INDEX IF NOT EXISTS ix_annual_report_documents_fiscal_year
    ON annual_report_documents (fiscal_year);

CREATE TABLE IF NOT EXISTS annual_financial_tables (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES annual_report_documents(id),
    chunk_index INTEGER NOT NULL,
    page_num INTEGER,
    section VARCHAR(255),
    table_kind VARCHAR(64) NOT NULL,
    raw_table_text TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_annual_financial_table_document_chunk
        UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_annual_financial_tables_document_id
    ON annual_financial_tables (document_id);
CREATE INDEX IF NOT EXISTS ix_annual_financial_tables_table_kind
    ON annual_financial_tables (table_kind);

CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(255) NOT NULL UNIQUE,
    aliases VARCHAR(512),
    statement_type VARCHAR(128),
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_financial_metrics_canonical_name
    ON financial_metrics (canonical_name);
CREATE INDEX IF NOT EXISTS ix_financial_metrics_statement_type
    ON financial_metrics (statement_type);

CREATE TABLE IF NOT EXISTS annual_financial_facts (
    id SERIAL PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES annual_financial_tables(id),
    metric_id INTEGER NOT NULL REFERENCES financial_metrics(id),
    row_index INTEGER NOT NULL DEFAULT 0,
    period_label VARCHAR(128),
    period_year INTEGER,
    period_type VARCHAR(64),
    value NUMERIC(24, 6),
    raw_value VARCHAR(128),
    unit VARCHAR(64),
    currency VARCHAR(32),
    raw_row TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_annual_financial_fact_source_metric
        UNIQUE (table_id, row_index, metric_id, period_label)
);

CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_table_id
    ON annual_financial_facts (table_id);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_metric_id
    ON annual_financial_facts (metric_id);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_period_label
    ON annual_financial_facts (period_label);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_period_year
    ON annual_financial_facts (period_year);

WITH source_companies AS (
    SELECT DISTINCT
        COALESCE(NULLIF(ticker, ''), lower(
            CASE
                WHEN position(' Annual Report' in title) > 0
                    THEN split_part(title, ' Annual Report', 1)
                ELSE title
            END
        )) AS company_key,
        CASE
            WHEN position(' Annual Report' in title) > 0
                THEN split_part(title, ' Annual Report', 1)
            ELSE title
        END AS name,
        NULLIF(ticker, '') AS ticker
    FROM annual_financial_facts_legacy
)
INSERT INTO financial_companies (company_key, name, ticker)
SELECT company_key, name, ticker
FROM source_companies
ON CONFLICT (company_key) DO UPDATE
SET name = EXCLUDED.name,
    ticker = EXCLUDED.ticker,
    updated_at = now();

WITH source_documents AS (
    SELECT DISTINCT
        l.doc_id,
        COALESCE(NULLIF(l.ticker, ''), lower(
            CASE
                WHEN position(' Annual Report' in l.title) > 0
                    THEN split_part(l.title, ' Annual Report', 1)
                ELSE l.title
            END
        )) AS company_key,
        l.title,
        l.fiscal_year,
        l.source
    FROM annual_financial_facts_legacy l
)
INSERT INTO annual_report_documents (doc_id, company_id, title, fiscal_year, source)
SELECT d.doc_id, c.id, d.title, d.fiscal_year, d.source
FROM source_documents d
JOIN financial_companies c ON c.company_key = d.company_key
ON CONFLICT (doc_id) DO UPDATE
SET company_id = EXCLUDED.company_id,
    title = EXCLUDED.title,
    fiscal_year = EXCLUDED.fiscal_year,
    source = EXCLUDED.source,
    updated_at = now();

INSERT INTO annual_financial_tables (
    document_id,
    chunk_index,
    page_num,
    section,
    table_kind,
    raw_table_text
)
SELECT DISTINCT
    d.id,
    l.chunk_index,
    l.page_num,
    l.section,
    l.table_kind,
    l.raw_table_text
FROM annual_financial_facts_legacy l
JOIN annual_report_documents d ON d.doc_id = l.doc_id
ON CONFLICT ON CONSTRAINT uq_annual_financial_table_document_chunk DO UPDATE
SET page_num = EXCLUDED.page_num,
    section = EXCLUDED.section,
    table_kind = EXCLUDED.table_kind,
    raw_table_text = EXCLUDED.raw_table_text,
    updated_at = now();

INSERT INTO financial_metrics (canonical_name, aliases, statement_type)
SELECT
    metric_name AS canonical_name,
    MAX(metric_alias) AS aliases,
    MAX(statement_type) AS statement_type
FROM annual_financial_facts_legacy
GROUP BY metric_name
ON CONFLICT (canonical_name) DO UPDATE
SET aliases = EXCLUDED.aliases,
    statement_type = EXCLUDED.statement_type,
    updated_at = now();

INSERT INTO annual_financial_facts (
    table_id,
    metric_id,
    row_index,
    period_label,
    period_year,
    period_type,
    value,
    raw_value,
    unit,
    currency,
    raw_row
)
SELECT
    t.id,
    m.id,
    l.row_index,
    l.period_label,
    l.period_year,
    l.period_type,
    l.value,
    l.raw_value,
    l.unit,
    l.currency,
    l.raw_row
FROM annual_financial_facts_legacy l
JOIN annual_report_documents d ON d.doc_id = l.doc_id
JOIN annual_financial_tables t
    ON t.document_id = d.id
    AND t.chunk_index = l.chunk_index
JOIN financial_metrics m ON m.canonical_name = l.metric_name
ON CONFLICT ON CONSTRAINT uq_annual_financial_fact_source_metric DO UPDATE
SET period_year = EXCLUDED.period_year,
    period_type = EXCLUDED.period_type,
    value = EXCLUDED.value,
    raw_value = EXCLUDED.raw_value,
    unit = EXCLUDED.unit,
    currency = EXCLUDED.currency,
    raw_row = EXCLUDED.raw_row,
    updated_at = now();

COMMIT;
