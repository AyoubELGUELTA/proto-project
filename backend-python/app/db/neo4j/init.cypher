// ============================================================
// NEO4J INITIALIZATION - GRAPH RAG
// ============================================================

// 1. CONSTRAINTS (UNICITÉ + PERFORMANCE)
// ============================================================

CREATE CONSTRAINT person_name IF NOT EXISTS
FOR (p:Person) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT event_name IF NOT EXISTS
FOR (e:Event) REQUIRE e.name IS UNIQUE;

CREATE CONSTRAINT place_name IF NOT EXISTS
FOR (pl:Place) REQUIRE pl.name IS UNIQUE;

CREATE CONSTRAINT concept_name IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT tribe_name IF NOT EXISTS
FOR (t:Tribe) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT period_name IF NOT EXISTS
FOR (p:Period) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT text_chunk_id IF NOT EXISTS
FOR (tc:TextChunk) REQUIRE tc.chunk_id IS UNIQUE;

CREATE CONSTRAINT community_id IF NOT EXISTS
FOR (com:Community) REQUIRE com.community_id IS UNIQUE;

// 2. INDEXES FULL-TEXT SEARCH (MULTILINGUE FR/AR/EN)
// ============================================================

CREATE FULLTEXT INDEX person_search IF NOT EXISTS
FOR (p:Person) ON EACH [p.name, p.arabic_name, p.aliases];

CREATE FULLTEXT INDEX event_search IF NOT EXISTS
FOR (e:Event) ON EACH [e.name, e.description];

CREATE FULLTEXT INDEX concept_search IF NOT EXISTS
FOR (c:Concept) ON EACH [c.name, c.domain, c.definition];

// 3. INDEXES PROPRIÉTÉS FRÉQUENTES
// ============================================================

CREATE INDEX person_type IF NOT EXISTS
FOR (p:Person) ON (p.type);

CREATE INDEX event_date IF NOT EXISTS
FOR (e:Event) ON (e.date);

CREATE INDEX chunk_doc IF NOT EXISTS
FOR (tc:TextChunk) ON (tc.doc_id);

CREATE INDEX community_level IF NOT EXISTS
FOR (com:Community) ON (com.level);

// 4. INITIALISATION COMPLÈTE
// ============================================================

// Log confirmation
CALL db.info() YIELD name, edition
RETURN 'Neo4j Graph RAG initialized: ' + name + ' (' + edition + ')' AS status;