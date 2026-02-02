import psycopg2
import json
import os 

def get_connection():
    """
    Retourne une connexion PostgreSQL.
    """
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "5432")

    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        conn.set_client_encoding("UTF8")
        return conn
    except Exception as e:
        print(f"❌ Erreur de connexion à la base de données : {e}")
        raise

def init_db():
    """
    Initialise la base de données : crée les tables avec support des headings.
    """
    conn = get_connection()
    cur = conn.cursor()
        
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    create_table_documents_query = """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        filename TEXT NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    create_table_chunks_query = """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_headings JSONB,                -- ✅ Hiérarchie complète ["Chapitre 1", "Section 1.1"]
        chunk_heading_full TEXT,              -- ✅ Titre complet "Chapitre 1 > Section 1.1"
        chunk_tables JSONB,
        chunk_images_base64 JSONB,
        chunk_page_numbers INTEGER[] DEFAULT '{}',
        chunk_type VARCHAR(20) DEFAULT 'content',
        is_identity BOOLEAN DEFAULT FALSE, -- Si ce chunk est la carte d'identité d'un document
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CHECK (chunk_type IN ('identity', 'content', 'toc')); -- Type de chunk: identity (fiche identité), content (contenu normal), toc (table des matières)
    CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_identity ON chunks(is_identity) WHERE is_identity = TRUE;
    
    -- ✅ Index pour recherche par titre (utile pour navigation)
    CREATE INDEX IF NOT EXISTS idx_chunks_heading
    ON chunks USING GIN (chunk_headings);
    """
    
    cur.execute(create_table_documents_query)
    cur.execute(create_table_chunks_query)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized (tables 'documents', 'chunks' with headings support).")

def get_or_create_document(filename):
    query = """
    INSERT INTO documents (filename) 
    VALUES (%s) 
    ON CONFLICT (filename) DO UPDATE SET filename = EXCLUDED.filename
    RETURNING doc_id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (filename,))
            doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id

def get_documents():
    query = "SELECT filename FROM documents ORDER BY created_at DESC;"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

def store_chunks_batch(analyzed_chunks, doc_id):
    """
    Stocke les chunks avec leurs métadonnées complètes (headings, pages, etc.)
    """
    insert_query = """
        INSERT INTO chunks (
            doc_id,
            chunk_index,
            chunk_text,
            chunk_headings,
            chunk_heading_full,
            chunk_tables,
            chunk_images_base64,
            chunk_page_numbers,
            chunk_type,
            is_identity
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING chunk_id;
    """

    chunk_ids = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get starting index
            cur.execute("""
                SELECT COALESCE(MAX(chunk_index), 0)
                FROM chunks
                WHERE doc_id = %s
            """, (doc_id,))
            start_index = cur.fetchone()[0]

            # Insert chunks ONE BY ONE (safe + ordered)
            for i, chunk in enumerate(analyzed_chunks):
                cur.execute(
                    insert_query,
                    (
                        doc_id,
                        start_index + i + 1,
                        chunk["text"],  # Texte seul (sans titre)
                        json.dumps(chunk["headings"]),  #  Hiérarchie ["Chapitre 1", "Section 1.1"]
                        chunk["heading_full"],  #  Titre complet "Chapitre 1 > Section 1.1"
                        json.dumps(chunk["tables"]),
                        json.dumps(chunk["images_base64"]),
                    )
                )
                chunk_ids.append(cur.fetchone()[0])

        conn.commit()

    return chunk_ids


def get_chunk_with_metadata(chunk_id):
    """
    Récupère un chunk avec toutes ses métadonnées (utile pour le retrieval)
    """
    query = """
    SELECT 
        chunk_id,
        chunk_text,
        chunk_headings,
        chunk_heading_full,
        chunk_tables,
        chunk_images_base64,
        chunk_page_numbers,
        chunk_type,
        is_identity
    FROM chunks
    WHERE chunk_id = %s;
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (chunk_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    "chunk_id": row[0],
                    "text": row[1],
                    "headings": json.loads(row[2]) if row[2] else [],
                    "heading_full": row[3],
                    "tables": json.loads(row[5]) if row[5] else [],
                    "images_base64": json.loads(row[6]) if row[6] else [],
                    "chunk_page_numbers": json.loads(row[7]) if row[7] else [],
                    "chunk_type": json.loads(row[8]) if row[8] else [],
                    "is_identity": json.loads(row[9]) if row[9] else []
                }
            return None