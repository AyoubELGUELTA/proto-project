from .base import get_connection, release_connection


async def init_db():
    """
    Initialise la base de données avec le support pgvector, 
    les entités, les liens, les tags hybrides et les co-occurrences.
    
    Architecture Hybrid:
    - Tags système (taxonomie ~50 thèmes) + tags auto-générés (flexibles)
    - Entity-centric avec co-occurrences pour relations
    - Normalisation des noms d'entités
    """
    conn = await get_connection()
    
    try:
        # ============================================================
        # 1. EXTENSIONS NÉCESSAIRES
        # ============================================================
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Extension pour normalisation texte (accents, etc.)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        print("✅ Extensions pgcrypto, pg_trgm, vector et unaccent activées")
        
        # ============================================================
        # 2. TABLE DOCUMENTS
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # ============================================================
        # 3. TABLE CHUNKS
        # ============================================================
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
                embedding vector(1024),
                processed_for_entities BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT check_chunk_type CHECK (chunk_type IN ('identity', 'content', 'toc'))
            );
        """)

        # ============================================================
        # 4. TABLE TAGS (SYSTÈME HYBRIDE)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                tag_id SERIAL PRIMARY KEY,
                label TEXT NOT NULL UNIQUE,
                tag_type VARCHAR(50) DEFAULT 'auto',  -- 'taxonomy' (système) ou 'auto' (généré)
                parent_id INTEGER REFERENCES tags(tag_id) ON DELETE SET NULL,
                description TEXT,
                is_system BOOLEAN DEFAULT FALSE,      -- Tag prédéfini vs auto-généré
                embedding vector(1024),               -- Pour matching sémantique
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ============================================================
        # 5. TABLE ENTITIES (ENRICHIE)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,                   -- Nom canonique (pas forcément unique si normalization fail)
                normalized_name TEXT NOT NULL UNIQUE, -- Version normalisée pour matching
                aliases TEXT[] DEFAULT '{}',          -- Toutes les variantes rencontrées
                normalized_aliases TEXT[],  
                entity_type VARCHAR(50),              -- PERSON, CONCEPT, EVENT, PLACE
                global_summary TEXT,                  -- Master chunk (résumé global)
                chunk_count INTEGER DEFAULT 0,        -- Nombre de chunks liés
                confidence_score FLOAT DEFAULT 1.0,   -- Confiance dans l'extraction
                last_updated TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # ============================================================
        # 6. TABLE ENTITY_LINKS (PIVOT ENTITÉS ↔ CHUNKS)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_links (
                link_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
                chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                relevance_score FLOAT DEFAULT 1.0,    -- Importance de l'entité dans ce chunk
                context_description TEXT,             -- Pourquoi ce lien ? (optionnel)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_id, chunk_id)           -- Pas de doublons
            );
        """)

        # ============================================================
        # 7. TABLE ENTITY_COOCCURRENCES (RELATIONS ENTRE ENTITÉS)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_cooccurrences (
                cooccurrence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_a_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
                entity_b_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
                co_occurrence_count INTEGER DEFAULT 1,
                shared_chunks UUID[] DEFAULT '{}',    -- Liste des chunks en commun
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (entity_a_id < entity_b_id),    -- Évite (A,B) et (B,A)
                UNIQUE(entity_a_id, entity_b_id)
            );
        """)

        # ============================================================
        # 8. TABLE ENTITY_TAGS (LIAISON ENTITÉS ↔ TAGS)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_tags (
                entity_id UUID REFERENCES entities(entity_id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
                confidence FLOAT DEFAULT 1.0,         -- Si assignation automatique
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (entity_id, tag_id)
            );
        """)

        # ============================================================
        # 9. TABLE CHUNK_TAGS (LIAISON CHUNKS ↔ TAGS)
        # ============================================================
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_tags (
                chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
                PRIMARY KEY (chunk_id, tag_id)
            );
        """)

        # ============================================================
        # 10. FONCTION NORMALISATION (SQL)
        # ============================================================
        # FONCTION DE NORMALISATION SQL
        # ============================================================
        await conn.execute("""
            CREATE OR REPLACE FUNCTION normalize_entity_name(text) 
            RETURNS text AS $$
            DECLARE
                result text;
            BEGIN
                -- 1. Strip + lowercase
                result := lower(trim($1));
                
                -- 2. Supprime accents
                result := unaccent(result);
                
                -- 3. Supprime apostrophes et quotes
                result := regexp_replace(result, '[''`´'']', '', 'g');
                
                -- 4. Supprime parenthèses et contenu
                result := regexp_replace(result, '\s*\([^)]*\)', '', 'g');
                
                -- 5. Supprime tirets et underscores
                result := regexp_replace(result, '[-_]', ' ', 'g');
                
                -- 6. Normalise "ibn", "bint", "al", "as"
                result := regexp_replace(result, '\mbin\M', 'ibn', 'g');
                result := regexp_replace(result, '\mal[\s-]', 'al ', 'g');
                result := regexp_replace(result, '\mas[\s-]', 'as ', 'g');
                
                -- 7. Supprime espaces multiples
                result := regexp_replace(result, '\s+', ' ', 'g');
                result := trim(result);
                
                RETURN result;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """)
        
        print("✅ Fonction normalize_entity_name() créée")

        # FONCTION INTERSECT DE TABLEAU, PRATIQUE POUR DEBOGGING....
        await conn.execute("""
            CREATE OR REPLACE FUNCTION array_intersect(anyarray, anyarray)
            RETURNS anyarray AS $$
                SELECT ARRAY(
                    SELECT unnest($1)
                    INTERSECT
                    SELECT unnest($2)
                )
            $$ LANGUAGE SQL IMMUTABLE;
        """)

        # ============================================================
        # 11. INDEXES PERFORMANCE
        # ============================================================
        
        # --- Chunks ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_processed ON chunks(processed_for_entities) 
            WHERE processed_for_entities = FALSE;
        """)
        
        # --- Tags ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_type ON tags(tag_type);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_system ON tags(is_system) 
            WHERE is_system = TRUE;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_embedding ON tags 
            USING hnsw (embedding vector_cosine_ops) 
            WHERE embedding IS NOT NULL;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_label_fts ON tags 
            USING gin(to_tsvector('french', label));
        """)
        
        # --- Entities ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_aliases ON entities USING GIN (aliases);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name_fts ON entities 
            USING gin(to_tsvector('french', name));
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_chunk_count ON entities(chunk_count DESC);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_normalized_aliases 
            ON entities USING GIN (normalized_aliases);
        """)
                
        # --- Entity Links ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_links_entity ON entity_links(entity_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_links_chunk ON entity_links(chunk_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_links_score ON entity_links(relevance_score DESC);
        """)
        
        # --- Entity Co-occurrences ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cooccur_entity_a ON entity_cooccurrences(entity_a_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cooccur_entity_b ON entity_cooccurrences(entity_b_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cooccur_count ON entity_cooccurrences(co_occurrence_count DESC);
        """)
        
        # --- Entity Tags ---
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_tags_entity ON entity_tags(entity_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_tags_tag ON entity_tags(tag_id);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_tags_confidence ON entity_tags(confidence DESC);
        """)
        
        print("\n" + "="*70)
        print("✅ Base de données ENTITY-CENTRIC HYBRID initialisée avec succès !")
        print("="*70)
        print("\nStructure créée :")
        print("  📦 Tables core : documents, chunks")
        print("  🏷️  Tags hybrides : système (taxonomie) + auto-générés")
        print("  🧬 Entities : avec normalisation + aliases")
        print("  🔗 Liens : entity_links, entity_cooccurrences, entity_tags")
        print("  ⚡ Indexes : HNSW (vector), GIN (arrays/FTS), B-tree (lookups)")
        print("  🛠️  Fonction SQL : normalize_entity_name()")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        raise
    finally:
        await release_connection(conn)


async def seed_system_tags():
    """
    Pré-remplit les tags système (taxonomie des ~50-100 thèmes religieux).
    À appeler UNE SEULE FOIS après init_db().
    """
    conn = await get_connection()
    
    try:
        # Liste de tes tags système (à adapter selon tes besoins)
        system_tags = [
            # Piliers & Fondamentaux
            ("Piliers de l'Islam", "Pratiques fondamentales obligatoires", None),
            ("Piliers de la Foi", "Croyances fondamentales", None),
            ("Tawhid (Unicité)", "Monothéisme et attributs divins", None),
            
            # Pratiques religieuses
            ("Salat (Prière)", "Prière rituelle et ses règles", "Piliers de l'Islam"),
            ("Zakat (Aumône)", "Aumône légale obligatoire", "Piliers de l'Islam"),
            ("Sawm (Jeûne)", "Jeûne du Ramadan et autres", "Piliers de l'Islam"),
            ("Hajj (Pèlerinage)", "Pèlerinage à La Mecque", "Piliers de l'Islam"),
            
            # Jurisprudence
            ("Fiqh (Jurisprudence)", "Science du droit islamique", None),
            ("Halal & Haram", "Licite et illicite", "Fiqh (Jurisprudence)"),
            ("Purification (Tahara)", "Ablutions et pureté rituelle", "Fiqh (Jurisprudence)"),
            
            # Histoire
            ("Sira (Vie du Prophète)", "Biographie prophétique", None),
            ("Mères des Croyants", "Épouses du Prophète", "Sira (Vie du Prophète)"),
            ("Compagnons (Sahaba)", "Compagnons du Prophète", "Sira (Vie du Prophète)"),
            ("Batailles & Événements", "Événements historiques majeurs", "Sira (Vie du Prophète)"),
            
            # Spiritualité
            ("Tassawuf (Soufisme)", "Dimension spirituelle et intérieure", None),
            ("Dhikr & Invocations", "Rappel d'Allah et prières", "Tassawuf (Soufisme)"),
            ("Comportement (Akhlaq)", "Éthique et moralité", None),
            
            # Sciences islamiques
            ("Tafsir (Exégèse)", "Interprétation du Coran", None),
            ("Hadith", "Paroles et actes prophétiques", None),
            ("Aqida (Croyance)", "Théologie islamique", None),
            
            # Ajoute les autres selon tes besoins...
        ]
        
        for label, description, parent_label in system_tags:
            # Trouve le parent_id si parent_label est fourni
            parent_id = None
            if parent_label:
                parent = await conn.fetchrow("""
                    SELECT tag_id FROM tags WHERE label = $1
                """, parent_label)
                if parent:
                    parent_id = parent['tag_id']
            
            # Insère le tag système
            await conn.execute("""
                INSERT INTO tags (label, tag_type, description, is_system, parent_id)
                VALUES ($1, 'taxonomy', $2, TRUE, $3)
                ON CONFLICT (label) DO NOTHING
            """, label, description, parent_id)
        
        count = await conn.fetchval("SELECT COUNT(*) FROM tags WHERE is_system = TRUE")
        print(f"\n✅ {count} tags système initialisés dans la taxonomie")
        
    except Exception as e:
        print(f"❌ Erreur lors du seeding des tags: {e}")
        raise
    finally:
        await release_connection(conn)