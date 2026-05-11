"""-- app/db/schema.sql (VERSION MULTI-USERS)

-- ============================================================
-- TABLE USERS (nouvelle pour multi-users)
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    username TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);  -- Login rapide

-- ============================================================
-- TABLE DOCUMENTS (modifiée pour multi-users)
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,  -- ✅ NOUVEAU
    filename TEXT NOT NULL,
    title TEXT,                    -- ✅ NOUVEAU : Titre du cours
    is_shared BOOLEAN DEFAULT FALSE,  -- ✅ NOUVEAU : Partage communautaire
    shared_at TIMESTAMP,           -- ✅ NOUVEAU : Date de partage
    upload_size_bytes BIGINT,      -- ✅ NOUVEAU : Taille du fichier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Contrainte : nom unique PAR utilisateur
    CONSTRAINT unique_filename_per_user UNIQUE (user_id, filename)
);

-- Index CRITIQUES pour multi-users
CREATE INDEX idx_docs_user_id ON documents(user_id);           -- Mes cours
CREATE INDEX idx_docs_shared ON documents(is_shared) WHERE is_shared = TRUE;  -- Cours partagés
CREATE INDEX idx_docs_created ON documents(created_at DESC);   -- Tri par date
CREATE INDEX idx_docs_user_created ON documents(user_id, created_at DESC);  -- Mes cours par date

-- ============================================================
-- TABLE CHUNKS (inchangée mais avec nouveaux index)
-- ============================================================
-- ... (ta table actuelle) ...

-- Index CRITIQUES pour scalabilité
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_doc_index ON chunks(doc_id, chunk_index);
CREATE INDEX idx_chunks_identity ON chunks(is_identity) WHERE is_identity = TRUE;
CREATE INDEX idx_chunks_type ON chunks(chunk_type);
CREATE INDEX idx_chunks_heading_gin ON chunks USING GIN (chunk_headings);

-- Index pour recherche full-text (optionnel mais recommandé)
CREATE INDEX idx_chunks_fulltext ON chunks USING GIN (to_tsvector('french', chunk_text));

-- ============================================================
-- TABLE SHARED_COURSES (pour gestion fine du partage)
-- ============================================================
CREATE TABLE IF NOT EXISTS shared_courses (
    share_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    shared_by_user_id UUID REFERENCES users(user_id),
    shared_with_user_id UUID REFERENCES users(user_id),  -- NULL = partage public
    permission VARCHAR(20) DEFAULT 'read',  -- read, write, admin
    shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT check_permission CHECK (permission IN ('read', 'write', 'admin'))
);

CREATE INDEX idx_shared_doc_id ON shared_courses(doc_id);
CREATE INDEX idx_shared_with ON shared_courses(shared_with_user_id);
CREATE INDEX idx_shared_by ON shared_courses(shared_by_user_id);"""

# TO ADD LATER IN THE INIT DB FUNCTION WHEN ILL THINK ABOUT USERS....