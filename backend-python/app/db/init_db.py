from .base import get_connection, release_connection
from app.db.entities import normalize_entity_name

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
                tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                label TEXT NOT NULL UNIQUE,
                tag_type VARCHAR(50) DEFAULT 'auto',
                parent_id UUID REFERENCES tags(tag_id) ON DELETE SET NULL, -- Correction type UUID
                description TEXT,
                is_system BOOLEAN DEFAULT FALSE,
                normalized_aliases TEXT[] DEFAULT ARRAY[]::TEXT[], -- La nouvelle colonne
                embedding vector(1024),
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
                tag_id UUID REFERENCES system_tags(tag_id) ON DELETE CASCADE,
                PRIMARY KEY (entity_id, tag_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        # Index GIN pour la recherche rapide dans les tableaux d'aliases (tags et entités)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_normalized_aliases ON tags USING GIN (normalized_aliases);
        """)
        
        # Index sur le label pour les recherches textuelles classiques
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_label_trgm ON tags USING gin (label gin_trgm_ops);
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
    Initialise la taxonomie système et normalise immédiatement les aliases.
    Utilise la logique de normalisation partagée pour garantir la cohérence.
    """
    conn = await get_connection()
    
    # Structure de base (le remplissage complet se fera au prochain prompt)
    system_tags = [
        # (Label, Description, Aliases, Parent_Label)
        
        ("Sira (Vie du Prophète)", "Biographie et vie du Prophète Muhammad (SAW)", ["Sirah", "السيرة النبوية", "Prophetic Biography"], None),
        ("Mères des Croyants", "Les 11 épouses du Prophète Muhammad (SAW)", ["Umm al-Mu'minin", "أمهات المؤمنين", "Mothers of Believers"], "Sira (Vie du Prophète)"),
        ("Compagnons (Sahaba)", "Compagnons du Prophète Muhammad (SAW) ayant vécu avec lui et cru en lui", ["Sahaba", "Sahabah", "أصحاب", "Companions"], "Sira (Vie du Prophète)"),
        ("Ahl al-Bayt", "Famille proche du Prophète : descendants de Fatima, Ali, et certains proches parents", ["Ahlul Bayt", "أهل البيت", "People of the House"], "Sira (Vie du Prophète)"),
        ("Enfants du Prophète", "Fils et filles du Prophète Muhammad (SAW)", ["أبناء النبي", "Children of Prophet"], "Sira (Vie du Prophète)"),
        ("Oncles et Tantes du Prophète", "Oncles paternels, maternels et tantes du Prophète", ["أعمام النبي", "Uncles of Prophet"], "Sira (Vie du Prophète)"),
        ("Batailles & Ghazawat", "Batailles et expéditions militaires dirigées par le Prophète", ["Ghazawat", "غزوات", "Battles", "Military Expeditions"], "Sira (Vie du Prophète)"),
        ("Saraya (Expéditions)", "Expéditions militaires envoyées par le Prophète sans sa présence directe", ["السرايا", "Sariyya"], "Sira (Vie du Prophète)"),
        ("Migrations (Hijra)", "Migrations historiques : Abyssinie, Médine", ["Hijrah", "الهجرة", "Migration"], "Sira (Vie du Prophète)"),
        ("Traités & Accords", "Traités de paix et accords conclus par le Prophète (Hudaybiya, etc.)", ["Treaties", "المعاهدات"], "Sira (Vie du Prophète)"),
        ("Miracles du Prophète", "Miracles et signes prophétiques (Isra wal Miraj, etc.)", ["Mu'jizat", "معجازات", "Prophetic Miracles"], "Sira (Vie du Prophète)"),
        ("Lieux de la Sira", "Lieux importants : Mecque, Médine, Badr, Uhud, etc.", ["أماكن السيرة", "Sira Locations"], "Sira (Vie du Prophète)"),

        # --- 2. AQIDA (Croyance) ---
        ("Aqida (Croyance)", "Théologie islamique et articles de foi", ["Aqeedah", "العقيدة", "Islamic Theology"], None),
        ("Piliers de la Foi", "Les 6 piliers de la foi (Iman) : Allah, Anges, Livres, Prophètes, Jour Dernier, Destin", ["Arkan al-Iman", "أركان الإيمان", "Pillars of Faith"], "Aqida (Croyance)"),
        ("Tawhid (Unicité)", "Monothéisme pur et attributs divins", ["التوحيد", "Oneness of Allah"], "Piliers de la Foi"),
        ("Noms d'Allah (Asma al-Husna)", "Les 99 noms et attributs d'Allah", ["Asma ul-Husna", "أسماء الله الحسنى", "99 Names"], "Piliers de la Foi"),
        ("Anges (Malaika)", "Anges en Islam : Jibril, Mikail, Israfil, Azrael, etc.", ["Mala'ika", "ملائكة", "Angels"], "Piliers de la Foi"),
        ("Livres Révélés", "Livres sacrés : Coran, Torah, Injil, Zabur, Suhuf", ["Kutub", "الكتب المقدسة", "Divine Books"], "Piliers de la Foi"),
        ("Prophètes et Messagers", "Prophètes mentionnés dans le Coran (25 nommés) et tradition", ["Anbiya", "أنبياء", "Rusul", "رسل", "Prophets"], "Piliers de la Foi"),
        ("Jour du Jugement (Yawm al-Qiyama)", "Résurrection, jugement dernier, fin des temps", ["Qiyamah", "يوم القيامة", "Day of Judgment"], "Piliers de la Foi"),
        ("Destin et Prédestination (Qadar)", "Concept du destin divin et libre arbitre", ["Qadar", "القدر", "Divine Decree"], "Piliers de la Foi"),
        ("Vie Après la Mort", "Barzakh, Paradis (Jannah), Enfer (Jahannam)", ["Akhira", "الآخرة", "Afterlife", "Jannah", "Jahannam"], "Piliers de la Foi"),
        ("Shirk et Kufr", "Polythéisme, mécréance et leurs formes", ["شرك", "كفر", "Polytheism"], "Aqida (Croyance)"),
        ("Sectes et Écoles Théologiques", "Ash'ari, Maturidi, Athari, etc.", ["المذاهب الكلامية", "Theological Schools"], "Aqida (Croyance)"),
        ("Signes de la Fin des Temps", "Signes mineurs et majeurs de l'Heure (Dajjal, Mahdi, Issa, Ya'juj Ma'juj)", ["أشراط الساعة", "Signs of Hour"], "Jour du Jugement (Yawm al-Qiyama)"),
        ("Attributs d'Allah", "Sifat : vie, connaissance, pouvoir, volonté, etc.", ["Sifat", "صفات الله", "Divine Attributes"], "Tawhid (Unicité)"),
        ("Bid'a et Innovations", "Innovations religieuses et leur statut", ["البدعة", "Religious Innovation"], "Aqida (Croyance)"),

        # --- 3. FIQH (Jurisprudence) ---
        ("Fiqh (Jurisprudence)", "Science du droit islamique et règles pratiques", ["الفقه", "Islamic Jurisprudence"], None),
        ("Piliers de l'Islam", "Les 5 piliers obligatoires : Shahada, Salat, Zakat, Sawm, Hajj", ["Arkan al-Islam", "أركان الإسلام", "Five Pillars"], "Fiqh (Jurisprudence)"),
        ("Shahada (Attestation de Foi)", "Témoignage de foi : La ilaha illa Allah, Muhammad Rasul Allah", ["الشهادة", "Testimony"], "Piliers de l'Islam"),
        ("Salat (Prière)", "Prière rituelle : 5 prières obligatoires, prières surérogatoires, règles", ["الصلاة", "Prayer", "Namaz"], "Piliers de l'Islam"),
        ("Zakat (Aumône Obligatoire)", "Aumône légale : taux, bénéficiaires, conditions", ["الزكاة", "Obligatory Charity"], "Piliers de l'Islam"),
        ("Sawm (Jeûne)", "Jeûne du Ramadan et jeûnes surérogatoires", ["الصوم", "Fasting", "Siyam"], "Piliers de l'Islam"),
        ("Hajj (Pèlerinage)", "Pèlerinage à La Mecque : rituels, conditions, types (Hajj, Umra)", ["الحج", "Pilgrimage"], "Piliers de l'Islam"),
        ("Purification (Tahara)", "Ablutions (wudu, ghusl), pureté rituelle, tayammum", ["الطهارة", "Wudu", "Ghusl", "Ablution"], "Fiqh (Jurisprudence)"),
        ("Halal & Haram", "Licite et illicite : nourriture, comportements, transactions", ["حلال", "حرام", "Lawful", "Forbidden"], "Fiqh (Jurisprudence)"),
        ("Nourriture et Boissons", "Règles alimentaires : viandes, abattage, alcool, etc.", ["الأطعمة", "Food Laws", "Dhabiha"], "Halal & Haram"),
        ("Muamalat (Transactions)", "Contrats, commerce, prêts, intérêts (riba)", ["المعاملات", "Islamic Finance", "Business"], "Fiqh (Jurisprudence)"),
        ("Riba (Intérêt/Usure)", "Interdiction de l'usure et intérêts", ["الربا", "Interest", "Usury"], "Muamalat (Transactions)"),
        ("Mariage (Nikah)", "Règles du mariage : conditions, droits, devoirs", ["النكاح", "Marriage", "Nikah"], "Fiqh (Jurisprudence)"),
        ("Divorce (Talaq)", "Divorce, répudiation, Khul, Iddah", ["الطلاق", "Divorce"], "Fiqh (Jurisprudence)"),
        ("Héritage (Mirath)", "Lois successorales islamiques", ["الميراث", "Inheritance"], "Fiqh (Jurisprudence)"),
        ("Qisas et Diyya", "Talion, prix du sang, justice pénale", ["القصاص", "الدية", "Retribution"], "Fiqh (Jurisprudence)"),
        ("Hudud (Peines Fixes)", "Peines coraniques fixes pour crimes majeurs", ["الحدود", "Fixed Punishments"], "Fiqh (Jurisprudence)"),
        ("Madhahib (Écoles Juridiques)", "Hanafi, Maliki, Shafi'i, Hanbali", ["المذاهب", "Schools of Law"], "Fiqh (Jurisprudence)"),
        ("Ijtihad et Taqlid", "Effort d'interprétation juridique vs suivi d'une école", ["الاجتهاد", "التقليد", "Independent Reasoning"], "Fiqh (Jurisprudence)"),
        ("Khilafa (Califat)", "Gouvernance islamique, leadership politique", ["الخلافة", "Caliphate"], "Fiqh (Jurisprudence)"),
        ("Jihad", "Effort spirituel et défense : règles, conditions, types", ["الجهاد", "Struggle"], "Fiqh (Jurisprudence)"),

        # --- 4. AKHLAQ (Éthique) ---
        ("Akhlaq (Éthique)", "Moralité, caractère, comportement islamique", ["الأخلاق", "Islamic Ethics", "Morality"], None),
        ("Vertus (Fadail)", "Patience (Sabr), gratitude (Shukr), sincérité (Ikhlas), humilité, générosité", ["الفضائل", "Virtues", "Sabr", "Shukr"], "Akhlaq (Éthique)"),
        ("Péchés Majeurs (Kaba'ir)", "Grands péchés : shirk, meurtre, fornication, etc.", ["الكبائر", "Major Sins"], "Akhlaq (Éthique)"),
        ("Relations Familiales", "Droits parents, enfants, famille, Birr al-Walidayn", ["Family Relations", "بر الوالدين"], "Akhlaq (Éthique)"),
        ("Relations Sociales", "Voisinage, fraternité, justice sociale", ["Social Relations", "العلاقات الاجتماعية"], "Akhlaq (Éthique)"),
        ("Adab (Bienséances)", "Étiquette islamique : salutations, repas, parole, habillement", ["الأدب", "Islamic Etiquette"], "Akhlaq (Éthique)"),
        ("Sincérité (Ikhlas)", "Pureté d'intention, agir pour Allah seul", ["الإخلاص", "Sincerity"], "Akhlaq (Éthique)"),
        ("Pardon et Miséricorde", "Afuw, Rahmah, clémence", ["العفو", "الرحمة", "Forgiveness"], "Akhlaq (Éthique)"),

        # --- 5. TASSAWUF (Spiritualité) ---
        ("Tassawuf (Soufisme)", "Dimension spirituelle et intérieure de l'Islam", ["التصوف", "Sufism", "Islamic Mysticism"], None),
        ("Dhikr (Rappel d'Allah)", "Invocations, formules de rappel, litanies", ["الذكر", "Remembrance of Allah"], "Tassawuf (Soufisme)"),
        ("Du'a (Invocations)", "Invocations prophétiques et supplications", ["الدعاء", "Supplication"], "Tassawuf (Soufisme)"),
        ("Ihsan (Excellence)", "Adorer Allah comme si tu Le voyais", ["الإحسان", "Spiritual Excellence"], "Tassawuf (Soufisme)"),
        ("Purification du Cœur (Tazkiya)", "Purification spirituelle, combat contre l'ego (nafs)", ["تزكية النفس", "Spiritual Purification", "Nafs"], "Tassawuf (Soufisme)"),
        ("Stations Spirituelles (Maqamat)", "Repentir, patience, confiance, amour divin", ["المقامات", "Spiritual Stations"], "Tassawuf (Soufisme)"),

        # --- 6. CORAN & TAFSIR ---
        ("Coran (Qur'an)", "Le Livre révélé : structure, sourates, versets", ["القرآن", "Quran", "Holy Quran"], None),
        ("Tafsir (Exégèse)", "Interprétation et commentaire du Coran", ["التفسير", "Quranic Exegesis"], "Coran (Qur'an)"),
        ("Sourates Mecquoises", "Sourates révélées à La Mecque", ["Makki Surahs", "السور المكية"], "Coran (Qur'an)"),
        ("Sourates Médinoises", "Sourates révélées à Médine", ["Madani Surahs", "السور المدنية"], "Coran (Qur'an)"),
        ("Sciences du Coran (Ulum al-Quran)", "Abrogation, occasions de révélation, récitation", ["علوم القرآن", "Quranic Sciences"], "Coran (Qur'an)"),
        ("Récitation (Tajwid)", "Règles de récitation coranique", ["التجويد", "Quranic Recitation"], "Coran (Qur'an)"),
        ("Versets Juridiques (Ayat al-Ahkam)", "Versets contenant des règles juridiques", ["آيات الأحكام", "Legal Verses"], "Coran (Qur'an)"),

        # --- 7. HADITH & SUNNA ---
        ("Hadith", "Paroles, actes et approbations du Prophète (SAW)", ["الحديث", "Prophetic Traditions"], None),
        ("Science du Hadith (Mustalah)", "Méthodologie : sahih, hasan, daif, isnad", ["مصطلح الحديث", "Hadith Sciences"], "Hadith"),
        ("Recueils de Hadith", "Sahih Bukhari, Muslim, Tirmidhi, Abu Dawud, etc.", ["الكتب الستة", "Six Books"], "Hadith"),
        ("Sunna (Tradition Prophétique)", "Pratique et exemple du Prophète", ["السنة", "Prophetic Tradition"], "Hadith"),
        ("Narrateurs de Hadith", "Compagnons et transmetteurs de hadith", ["رواة الحديث", "Hadith Narrators"], "Hadith"),

        # --- 8. HISTOIRE ISLAMIQUE ---
        ("Histoire Islamique", "Événements historiques post-prophétiques", ["التاريخ الإسلامي", "Islamic History"], None),
        ("Califes Bien-Guidés (Khulafa Rashidun)", "Abu Bakr, Umar, Uthman, Ali", ["الخلفاء الراشدون", "Rightly Guided Caliphs"], "Histoire Islamique"),
        ("Dynasties et Empires", "Omeyyades, Abbassides, Ottomans, etc.", ["الدول الإسلامية", "Islamic Empires"], "Histoire Islamique"),
        ("Savants et Érudits", "Grands imams, muftis, muhaddithun", ["العلماء", "Islamic Scholars"], "Histoire Islamique"),
        # --- 9. PERSONNAGES SPÉCIFIQUES & ÉRUDITION FÉMININE ---
        ("Femmes Érudites & Savantes", "Femmes ayant marqué l'histoire de l'islam par leur savoir (Hadith, Fiqh, Poésie)", ["Muhaddithat", "Al-Muhaddithat", "Femmes savantes", "Scholar women", "عالمات"], "Histoire Islamique"),
    ]
    
    
    try:
        print("🌱 [SEEDING] Début de l'initialisation des tags...")

        for label, description, aliases, parent_label in system_tags:
            # 1. Résolution du Parent
            parent_id = None
            if parent_label:
                parent_row = await conn.fetchrow("SELECT tag_id FROM tags WHERE label = $1", parent_label)
                if parent_row:
                    parent_id = parent_row['tag_id']

            # 2. Logique de ton script : Normalisation des aliases + Label
            normalized = []
            if aliases:
                for alias in aliases:
                    norm = normalize_entity_name(alias)
                    if norm:
                        normalized.append(norm)
            
            label_norm = normalize_entity_name(label)
            if label_norm and label_norm not in normalized:
                normalized.insert(0, label_norm)

            # 3. Insertion / Mise à jour atomique
            await conn.execute("""
                INSERT INTO tags (label, tag_type, description, is_system, aliases, normalized_aliases, parent_id)
                VALUES ($1, 'taxonomy', $2, TRUE, $3, $4, $5)
                ON CONFLICT (label) DO UPDATE 
                SET description = EXCLUDED.description,
                    aliases = EXCLUDED.aliases,
                    normalized_aliases = EXCLUDED.normalized_aliases,
                    parent_id = EXCLUDED.parent_id
            """, label, description, aliases, normalized, parent_id)

        count = await conn.fetchval("SELECT COUNT(*) FROM tags WHERE is_system = TRUE")
        print(f"✅ [SEEDING] Terminé : {count} tags système prêts et normalisés.")

    except Exception as e:
        print(f"❌ [SEEDING] Erreur : {e}")
        raise
    finally:
        await release_connection(conn)