import psycopg2
import json
import os 
from typing import List, Dict, Any, Optional
import uuid

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
    Initialise la base de données : crée les tables avec support complet
    des headings, page numbers, et chunks identité.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # 1. Activer l'extension pgcrypto pour gen_random_uuid()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        print("✅ Extension pgcrypto activée")
        
        # 2. Créer la table documents
        create_table_documents_query = """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_documents_query)
        print("✅ Table 'documents' créée/vérifiée")
        
        # 3. Créer la table chunks avec toutes les colonnes
        create_table_chunks_query = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_headings JSONB,              -- Hiérarchie complète ["Chapitre 1", "Section 1.1"]
            chunk_heading_full TEXT,            -- Titre complet "Chapitre 1 > Section 1.1"
            chunk_page_numbers INTEGER[] DEFAULT '{}',  -- Liste des numéros de pages
            chunk_tables JSONB DEFAULT '[]',    -- Tables extraites (Markdown)
            chunk_images_base64 JSONB DEFAULT '[]',  -- Images extraites (base64)
            chunk_type VARCHAR(20) DEFAULT 'content',  -- Type: identity/content/toc
            is_identity BOOLEAN DEFAULT FALSE,  -- Flag pour chunk identité
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Contrainte sur chunk_type
            CONSTRAINT check_chunk_type CHECK (chunk_type IN ('identity', 'content', 'toc'))
        );
        """
        cur.execute(create_table_chunks_query)
        print("✅ Table 'chunks' créée/vérifiée")
        
        # 4. Créer les index pour optimiser les requêtes
        create_indexes_query = """
        -- Index sur chunk_type pour filtrer rapidement par type
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
        
        -- Index sur is_identity pour récupérer rapidement les fiches identité
        CREATE INDEX IF NOT EXISTS idx_chunks_identity ON chunks(is_identity) 
        WHERE is_identity = TRUE;
        
        -- Index GIN sur chunk_headings pour recherche full-text dans les titres
        CREATE INDEX IF NOT EXISTS idx_chunks_heading_gin 
        ON chunks USING GIN (chunk_headings);
        
        -- Index sur doc_id pour jointures rapides
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
        
        -- Index composite pour requêtes fréquentes (doc_id + chunk_index)
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_index 
        ON chunks(doc_id, chunk_index);
        """
        cur.execute(create_indexes_query)
        print("✅ Index créés/vérifiés")
        
        # 5. Ajouter des commentaires pour la documentation
        add_comments_query = """
        COMMENT ON TABLE documents IS 
        'Table principale stockant les métadonnées des documents ingérés';
        
        COMMENT ON TABLE chunks IS 
        'Chunks de texte extraits des documents avec métadonnées enrichies';
        
        COMMENT ON COLUMN chunks.chunk_headings IS 
        'Hiérarchie des titres au format JSON array, ex: ["Chapitre 1", "Section 1.1"]';
        
        COMMENT ON COLUMN chunks.chunk_heading_full IS 
        'Titre complet formaté, ex: "Chapitre 1 > Section 1.1"';
        
        COMMENT ON COLUMN chunks.chunk_page_numbers IS 
        'Liste des numéros de pages couvertes par ce chunk';
        
        COMMENT ON COLUMN chunks.chunk_type IS 
        'Type de chunk: identity (fiche identité), content (contenu normal), toc (table des matières)';
        
        COMMENT ON COLUMN chunks.is_identity IS 
        'TRUE si ce chunk est une fiche identité générée automatiquement pour le document';
        
        COMMENT ON COLUMN chunks.chunk_tables IS 
        'Tables extraites du chunk au format JSON (Markdown)';
        
        COMMENT ON COLUMN chunks.chunk_images_base64 IS 
        'Images extraites du chunk au format JSON (base64)';
        """
        cur.execute(add_comments_query)
        print("✅ Commentaires ajoutés")
        
        # 6. Commit final
        conn.commit()
        print("\n" + "="*60)
        print("✅ Base de données initialisée avec succès !")
        print("="*60)
        print("📋 Tables créées:")
        print("   - documents (doc_id, filename, created_at)")
        print("   - chunks (avec headings, page_numbers, chunk_type, is_identity)")
        print("📊 Index créés:")
        print("   - idx_chunks_type")
        print("   - idx_chunks_identity")
        print("   - idx_chunks_heading_gin")
        print("   - idx_chunks_doc_id")
        print("   - idx_chunks_doc_index")
        print("="*60)
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Erreur lors de l'initialisation de la base de données: {e}")
        raise
    
    finally:
        cur.close()
        conn.close()

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

async def store_chunks_batch(chunks: List[Dict[str, Any]], doc_id: str) -> List[str]:
    """
    Stocke un batch de chunks dans PostgreSQL.
    
    Args:
        chunks: Liste de dicts contenant les données des chunks
        doc_id: UUID du document parent
    
    Returns:
        Liste des chunk_ids créés
    """
    conn = await get_connection()
    chunk_ids = []
    
    try:
        async with conn.cursor() as cursor:
            for i, chunk_data in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                await cursor.execute("""
                    INSERT INTO chunks (
                        chunk_id,
                        doc_id,
                        chunk_index,
                        chunk_text,
                        chunk_headings,
                        chunk_heading_full,
                        chunk_page_numbers, 
                        chunk_tables,
                        chunk_images_base64,
                        chunk_type,       
                        is_identity      
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    chunk_id,
                    doc_id,
                    chunk_data.get("chunk_index", i),
                    chunk_data.get("chunk_text", ""),
                    json.dumps(chunk_data.get("chunk_headings", [])),
                    chunk_data.get("chunk_heading_full", ""),
                    chunk_data.get("chunk_page_numbers", []), 
                    json.dumps(chunk_data.get("chunk_tables", [])),
                    json.dumps(chunk_data.get("chunk_images_base64", [])),
                    chunk_data.get("chunk_type", "content"),  
                    chunk_data.get("is_identity", False)      
                ))
                
                chunk_ids.append(chunk_id)
        
        await conn.commit()
        print(f"✅ {len(chunk_ids)} chunks stockés dans PostgreSQL")
        
    except Exception as e:
        await conn.rollback()
        print(f"❌ Erreur lors du stockage des chunks : {e}")
        raise
    finally:
        await conn.close()
    
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
        

async def store_identity_chunk(
    doc_id: str,
    identity_text: str,
    pages_sampled: List[int]
) -> str:
    """
    Stocke le chunk identité d'un document.
    
    Args:
        doc_id: UUID du document
        identity_text: Texte de la fiche identité
        pages_sampled: Pages utilisées pour générer la fiche
    
    Returns:
        chunk_id du chunk identité créé
    """
    chunk_id = str(uuid.uuid4())
    conn = await get_connection()
    
    try:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO chunks (
                    chunk_id,
                    doc_id,
                    chunk_index,
                    chunk_text,
                    chunk_headings,
                    chunk_heading_full,
                    chunk_page_numbers,
                    chunk_tables,
                    chunk_images_base64,
                    chunk_type,
                    is_identity
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                chunk_id,
                doc_id,
                -1,  # Index spécial pour chunk identité
                identity_text,
                json.dumps(["DOCUMENT_IDENTITY"]),
                "DOCUMENT_IDENTITY",
                pages_sampled,
                json.dumps([]),
                json.dumps([]),
                'identity',  # Type spécial
                True         # Flag identité
            ))
        
        await conn.commit()
        print(f"✅ Chunk identité stocké : {chunk_id}")
        
    except Exception as e:
        await conn.rollback()
        print(f"❌ Erreur lors du stockage du chunk identité : {e}")
        raise
    finally:
        await conn.close()
    
    return chunk_id
