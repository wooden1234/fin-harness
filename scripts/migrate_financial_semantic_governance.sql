BEGIN;

CREATE TABLE IF NOT EXISTS fin_core.canonical_metrics (
    code VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    statement_type VARCHAR(64),
    value_type VARCHAR(32) NOT NULL DEFAULT 'amount',
    default_unit VARCHAR(64),
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_canonical_metrics_name
    ON fin_core.canonical_metrics (name);
CREATE INDEX IF NOT EXISTS ix_canonical_metrics_statement_type
    ON fin_core.canonical_metrics (statement_type);

CREATE TABLE IF NOT EXISTS fin_core.canonical_metric_aliases (
    id SERIAL PRIMARY KEY,
    canonical_code VARCHAR(64) NOT NULL REFERENCES fin_core.canonical_metrics(code),
    alias VARCHAR(255) NOT NULL,
    normalized_alias VARCHAR(255) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'seed',
    priority INTEGER NOT NULL DEFAULT 100,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_canonical_metric_alias UNIQUE (alias)
);

CREATE INDEX IF NOT EXISTS ix_canonical_metric_aliases_canonical_code
    ON fin_core.canonical_metric_aliases (canonical_code);
CREATE INDEX IF NOT EXISTS ix_canonical_metric_aliases_alias
    ON fin_core.canonical_metric_aliases (alias);
CREATE INDEX IF NOT EXISTS ix_canonical_metric_aliases_normalized_alias
    ON fin_core.canonical_metric_aliases (normalized_alias);

CREATE TABLE IF NOT EXISTS fin_core.company_metric_mappings (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES fin_core.financial_companies(id),
    canonical_code VARCHAR(64) NOT NULL REFERENCES fin_core.canonical_metrics(code),
    source_metric_id INTEGER NOT NULL REFERENCES fin_core.financial_metrics(id),
    source_metric_name VARCHAR(255) NOT NULL,
    statement_type VARCHAR(128),
    valid_from_year INTEGER,
    valid_to_year INTEGER,
    priority INTEGER NOT NULL DEFAULT 100,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    mapping_source VARCHAR(32) NOT NULL DEFAULT 'seed',
    review_status VARCHAR(32) NOT NULL DEFAULT 'approved',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_company_metric_mapping_source
        UNIQUE (company_id, canonical_code, source_metric_id)
);

CREATE INDEX IF NOT EXISTS ix_company_metric_mappings_company_id
    ON fin_core.company_metric_mappings (company_id);
CREATE INDEX IF NOT EXISTS ix_company_metric_mappings_canonical_code
    ON fin_core.company_metric_mappings (canonical_code);
CREATE INDEX IF NOT EXISTS ix_company_metric_mappings_source_metric_id
    ON fin_core.company_metric_mappings (source_metric_id);
CREATE INDEX IF NOT EXISTS ix_company_metric_mappings_review_status
    ON fin_core.company_metric_mappings (review_status);

CREATE TABLE IF NOT EXISTS fin_core.raw_table_cells (
    id SERIAL PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES fin_core.annual_financial_tables(id),
    document_id INTEGER NOT NULL REFERENCES fin_core.annual_report_documents(id),
    page_num INTEGER,
    row_index INTEGER NOT NULL,
    col_index INTEGER NOT NULL,
    row_header VARCHAR(512),
    col_header VARCHAR(512),
    cell_text TEXT,
    normalized_value NUMERIC(24, 6),
    unit VARCHAR(64),
    currency VARCHAR(32),
    bbox_json TEXT,
    extractor VARCHAR(64) NOT NULL DEFAULT 'pdf_pipeline',
    extract_version VARCHAR(64) NOT NULL DEFAULT 'v1',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_raw_table_cell_position
        UNIQUE (table_id, row_index, col_index)
);

CREATE INDEX IF NOT EXISTS ix_raw_table_cells_table_id
    ON fin_core.raw_table_cells (table_id);
CREATE INDEX IF NOT EXISTS ix_raw_table_cells_document_id
    ON fin_core.raw_table_cells (document_id);

ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS canonical_code VARCHAR(64);
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS source_cell_id INTEGER;
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS quality_status VARCHAR(32) NOT NULL DEFAULT 'pending';
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(32) NOT NULL DEFAULT 'unreviewed';
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS validation_errors TEXT;
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS extract_version VARCHAR(64) NOT NULL DEFAULT 'v1';
ALTER TABLE fin_core.annual_financial_facts
    ADD COLUMN IF NOT EXISTS is_published BOOLEAN NOT NULL DEFAULT false;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_annual_financial_facts_canonical_code'
    ) THEN
        ALTER TABLE fin_core.annual_financial_facts
            ADD CONSTRAINT fk_annual_financial_facts_canonical_code
            FOREIGN KEY (canonical_code)
            REFERENCES fin_core.canonical_metrics(code);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_annual_financial_facts_source_cell_id'
    ) THEN
        ALTER TABLE fin_core.annual_financial_facts
            ADD CONSTRAINT fk_annual_financial_facts_source_cell_id
            FOREIGN KEY (source_cell_id)
            REFERENCES fin_core.raw_table_cells(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_canonical_code
    ON fin_core.annual_financial_facts (canonical_code);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_source_cell_id
    ON fin_core.annual_financial_facts (source_cell_id);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_quality_status
    ON fin_core.annual_financial_facts (quality_status);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_review_status
    ON fin_core.annual_financial_facts (review_status);
CREATE INDEX IF NOT EXISTS ix_annual_financial_facts_is_published
    ON fin_core.annual_financial_facts (is_published);

COMMIT;
