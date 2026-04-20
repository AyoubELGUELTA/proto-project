
CREATE_SCHEMA_QUERY = """
-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Table Documents
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL UNIQUE,
    metadata JSONB DEFAULT '{}',
    status TEXT DEFAULT 'PENDING', 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_status CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

-- Table Chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT DEFAULT 'CONTENT',
    chunk_text TEXT NOT NULL,
    chunk_headings JSONB DEFAULT '[]',
    chunk_heading_full TEXT,
    chunk_page_numbers INTEGER[] DEFAULT '{}',
    chunk_tables JSONB DEFAULT '[]',
    chunk_images_urls TEXT[] DEFAULT '{}',
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_chunk_type CHECK (chunk_type IN ('IDENTITY', 'CONTENT', 'TOC'))
);

CREATE TABLE IF NOT EXISTS encyclopedia (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- UUID unique
    slug TEXT NOT NULL UNIQUE,                      -- Identifiant métier (ex: 'UMAR_IBN_AL_KHATTAB')
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    category TEXT NOT NULL,
    core_summary TEXT NOT NULL,
    properties JSONB DEFAULT '{}',                  -- Contient les alias, nasab, etc.
    is_verified BOOLEAN DEFAULT TRUE,
    review_status TEXT DEFAULT 'PENDING',
    last_updated_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_review_status CHECK (review_status IN ('NOT_KNOWN', 'PENDING', 'LLM_VALIDATED', 'CORE_VALIDATED', 'OFFICIAL'))
);



CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_encyclopedia_slug ON encyclopedia(slug);
CREATE INDEX IF NOT EXISTS idx_encyclopedia_properties_gin ON encyclopedia USING GIN (properties);
CREATE INDEX IF NOT EXISTS idx_encyclopedia_verification ON encyclopedia(is_verified);

"""