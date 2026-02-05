import asyncpg
import json
import os 
from typing import List, Dict, Any, Optional
import uuid

# Pool de connexions global
_pool = None

async def init_pool():
    """Initialise le pool de connexions async"""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            min_size=5,
            max_size=20
        )
        print("✅ Pool PostgreSQL initialisé (async)")
    return _pool


async def get_connection():
    """Récupère une connexion depuis le pool"""
    pool = await init_pool()
    return await pool.acquire()


async def release_connection(conn):
    """Libère une connexion vers le pool"""
    pool = await init_pool()
    await pool.release(conn)


async def init_db():
    """
    Initialise la base de données : crée les tables avec support complet
    des headings, page numbers, et chunks identité.
    """
    conn = await get_connection()
    
    try:
        # 1. Activer l'extension pgcrypto pour gen_random_uuid()
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        print("✅ Extension pgcrypto activée")
        
        # 2. Créer la table documents
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✅ Table 'documents' créée/vérifiée")
        
        # 3. Créer la table chunks avec toutes les colonnes
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_visual_summary TEXT DEFAULT '', -- Stocke le visual summary si image ou tableau
                chunk_headings JSONB,
                chunk_heading_full TEXT,
                chunk_page_numbers INTEGER[] DEFAULT '{}',
                chunk_tables JSONB DEFAULT '[]',
                chunk_images_urls TEXT[] DEFAULT '{}',  
                chunk_type VARCHAR(20) DEFAULT 'content',
                is_identity BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CONSTRAINT check_chunk_type CHECK (chunk_type IN ('identity', 'content', 'toc'))
            );
        """)
        print("✅ Table 'chunks' créée/vérifiée")
        
        # 4. Créer les index pour optimiser les requêtes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_identity ON chunks(is_identity) 
            WHERE is_identity = TRUE;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_heading_gin 
            ON chunks USING GIN (chunk_headings);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_index 
            ON chunks(doc_id, chunk_index);
        """)
        print("✅ Index créés/vérifiés")
        
        # 5. Ajouter des commentaires pour la documentation
        await conn.execute("""
            COMMENT ON TABLE documents IS 
            'Table principale stockant les métadonnées des documents ingérés';
        """)
        await conn.execute("""
            COMMENT ON TABLE chunks IS 
            'Chunks de texte extraits des documents avec métadonnées enrichies';
        """)
        # ... (autres commentaires si nécessaire)
        
        print("\n" + "="*60)
        print("✅ Base de données initialisée avec succès !")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de la base de données: {e}")
        raise
    finally:
        await release_connection(conn)


async def get_or_create_document(filename: str) -> str:
    """Récupère ou crée un document"""
    conn = await get_connection()
    try:
        # ✅ $1 au lieu de %s
        doc_id = await conn.fetchval("""
            INSERT INTO documents (filename) 
            VALUES ($1) 
            ON CONFLICT (filename) DO UPDATE SET filename = EXCLUDED.filename
            RETURNING doc_id;
        """, filename)
        return str(doc_id)
    finally:
        await release_connection(conn)


async def get_documents() -> List[str]:
    """Récupère la liste des documents"""
    conn = await get_connection()
    try:
        rows = await conn.fetch(
            "SELECT filename FROM documents ORDER BY created_at DESC;"
        )
        return [row['filename'] for row in rows]
    finally:
        await release_connection(conn)


async def store_chunks_batch(chunks: List[Dict[str, Any]], doc_id: str) -> List[str]:
    """
    Stocke un batch de chunks dans PostgreSQL.
    """
    conn = await get_connection()
    chunk_ids = []
    
    try:
        # ✅ Utiliser une transaction explicite
        async with conn.transaction():
            for i, chunk_data in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                await conn.execute("""
                    INSERT INTO chunks (
                        chunk_id, doc_id, chunk_index, chunk_text, 
                        chunk_visual_summary, chunk_headings, chunk_heading_full, 
                        chunk_page_numbers, chunk_tables, chunk_images_urls, 
                        chunk_type, is_identity
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    chunk_id,
                    doc_id,
                    chunk_data.get("chunk_index", i),
                    chunk_data.get("text", ""),
                    chunk_data.get("visual_summary", ""), 
                    json.dumps(chunk_data.get("headings", [])),
                    chunk_data.get("heading_full", ""),
                    chunk_data.get("page_numbers", []), 
                    json.dumps(chunk_data.get("tables", [])),
                    chunk_data.get("images_urls", []),                    
                    chunk_data.get("chunk_type", "content"),  
                    chunk_data.get("is_identity", False)
                )
                chunk_ids.append(chunk_id)
        
        print(f"✅ {len(chunk_ids)} chunks stockés dans PostgreSQL")
        
    except Exception as e:
        print(f"❌ Erreur lors du stockage des chunks : {e}")
        raise
    finally:
        await release_connection(conn)
    
    return chunk_ids


async def get_chunk_with_metadata(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Récupère un chunk avec toutes ses métadonnées
    """
    conn = await get_connection()
    try:
        row = await conn.fetchrow("""
            SELECT chunk_id, chunk_text, chunk_visual_summary, chunk_headings, 
                   chunk_heading_full, chunk_tables, chunk_images_urls, 
                   chunk_page_numbers, chunk_type, is_identity
            FROM chunks WHERE chunk_id = $1;
        """, chunk_id)
        
        if row:
            return {
                "chunk_id": str(row['chunk_id']),
                "text": row['chunk_text'],
                "visual_summary": row['chunk_visual_summary'], 
                "headings": row['chunk_headings'] if row['chunk_headings'] else [],
                "heading_full": row['chunk_heading_full'],
                "tables": row['chunk_tables'] if row['chunk_tables'] else [],
                "images_urls": row['chunk_images_urls'] if row['chunk_images_urls'] else [],
                "chunk_page_numbers": row['chunk_page_numbers'] if row['chunk_page_numbers'] else [],
                "chunk_type": row['chunk_type'],
                "is_identity": row['is_identity']
            }
        return None
    finally:
        await release_connection(conn)


async def store_identity_chunk(
    doc_id: str,
    identity_text: str,
    pages_sampled: List[int]
) -> str:
    """
    Stocke le chunk identité d'un document.
    """
    chunk_id = str(uuid.uuid4())
    conn = await get_connection()
    
    try:
        await conn.execute("""
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, chunk_text, 
                chunk_visual_summary, chunk_headings, chunk_heading_full, 
                chunk_page_numbers, chunk_tables, chunk_images_urls, 
                chunk_type, is_identity
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            chunk_id, doc_id, -1, identity_text,
            "", # visual_summary vide pour l'identité, pas d'image ou de tableau en principe
            json.dumps(["DOCUMENT_IDENTITY"]), "DOCUMENT_IDENTITY",
            pages_sampled, json.dumps([]), [], 'identity', True
        )
        
        print(f"✅ Chunk identité stocké : {chunk_id}")
        
    except Exception as e:
        print(f"❌ Erreur lors du stockage du chunk identité : {e}")
        raise
    finally:
        await release_connection(conn)
    
    return chunk_id

async def fetch_identities_by_doc_ids(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Récupère les chunks identité pour une liste de doc_ids donnés.
    Utile pour injecter le contexte global après le reranking.
    """
    if not doc_ids:
        return []

    # Conversion des strings en UUID si nécessaire pour asyncpg
    uuid_list = [uuid.UUID(d) if isinstance(d, str) else d for d in doc_ids]
    
    conn = await get_connection()
    try:
        # On récupère l'essentiel pour le LLM
        rows = await conn.fetch("""
            SELECT 
                chunk_id, 
                doc_id, 
                chunk_text, 
                chunk_visual_summary,
                chunk_heading_full,
                is_identity
            FROM chunks
            WHERE doc_id = ANY($1::uuid[]) AND is_identity = TRUE;
        """, uuid_list)
        
        identities = []
        for row in rows:
            identities.append({
                "chunk_id": str(row['chunk_id']),
                "doc_id": str(row['doc_id']),
                "text": row['chunk_text'],
                "visual_summary": row['chunk_visual_summary'],
                "heading_full": row['chunk_heading_full'],
                "is_identity": True,
                "chunk_type": "identity",
                "vector_score": 1.0  # On lui donne un score fictif parfait car c'est le contexte maître
            })
        return identities
    finally:
        await release_connection(conn)

async def update_chunks_with_ai_data(summarised_chunks: List[Dict[str, Any]]):
    """
    Met à jour les chunks existants avec le texte enrichi et le visual summary.
    """
    conn = await get_connection()
    try:
        async with conn.transaction():
            for chunk in summarised_chunks:
                await conn.execute("""
                    UPDATE chunks 
                    SET chunk_text = $1, 
                        chunk_visual_summary = $2
                    WHERE chunk_id = $3::uuid
                """, 
                chunk["text"], 
                chunk["visual_summary"], 
                chunk["chunk_id"])
        print(f"✅ {len(summarised_chunks)} chunks enrichis par l'IA mis à jour en BDD.")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour AI des chunks : {e}")
        raise
    finally:
        await release_connection(conn)