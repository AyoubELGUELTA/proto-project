from .base import get_connection, release_connection


async def init_db():
    """
    Initialise la base de données avec le support pgvector, 
    les entités, les liens et la taxonomie (tags).
    """
    conn = await get_connection()
    
    try:
        # 1. Extensions nécessaires
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        # ✅ IMPORTANT : Activation de pgvector pour stocker nos embeddings
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ Extensions pgcrypto et vector activées")
        
        # 2. Table documents
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 3. Table chunks (Enrichie avec la colonne embedding)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_visual_summary TEXT DEFAULT '',
                chunk_headings JSONB,
                chunk_heading_full TEXT,
                chunk_page_numbers INTEGER[] DEFAULT '{}',
                chunk_tables JSONB DEFAULT '[]',
                chunk_images_urls TEXT[] DEFAULT '{}',  
                chunk_type VARCHAR(20) DEFAULT 'content',
                is_identity BOOLEAN DEFAULT FALSE,
                -- ✅ Ajout du vecteur (1024 pour BGE-M3 local, mais 1500 et quelques pour OpenAI text-embedding-3-small, etc.. A ADAPTER)
                embedding vector(1024), 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT check_chunk_type CHECK (chunk_type IN ('identity', 'content', 'toc'))
            );
        """)

        # 4. Table entities (Le coeur de ton nouveau système)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL UNIQUE,
                aliases TEXT[] DEFAULT '{}', --  Variantes (Wudu, Woudou, etc.)
                entity_type VARCHAR(50),      -- PERSONNE, LIEU, CONCEPT, etc.
                global_summary TEXT,          -- Résumé global de l'entité
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 5. Table entity_links (La table pivot pour le Graph)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_links (
                link_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
                chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                relevance_score FLOAT DEFAULT 1.0, -- Score d'importance dans ce chunk
                context_description TEXT,          -- Pourquoi ce lien ? (Optionnel)
                UNIQUE(entity_id, chunk_id)        -- Évite les doublons de liens
            );
        """)

        # 6. Système de Tags (Taxonomie flexible)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                tag_id SERIAL PRIMARY KEY,
                label TEXT NOT NULL UNIQUE,
                parent_id INTEGER REFERENCES tags(tag_id) ON DELETE SET NULL
            );
            
            CREATE TABLE IF NOT EXISTS chunk_tags (
                chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
                PRIMARY KEY (chunk_id, tag_id)
            );
        """)

        # 7. Indexation Performance
        # Index HNSW pour la recherche vectorielle (ultra rapide)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        # Index GIN pour la recherche dans les tableaux d'aliases
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_aliases ON entities USING GIN (aliases);
        """)
        
        print("\n" + "="*60)
        print("✅ Base de données ENTITY-CENTRIC initialisée avec pgvector !")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        raise
    finally:
        await release_connection(conn)