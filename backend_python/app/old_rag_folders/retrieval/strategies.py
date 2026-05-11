from typing import List, Dict, Any
from abc import ABC, abstractmethod
from app.db.base import get_connection, release_connection
import uuid


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTION - Formatage des chunks
# ═══════════════════════════════════════════════════════════

def format_chunk_for_context(chunk_row) -> Dict:
    """
    Transforme un row PostgreSQL en format contexte compatible.
    Reconstitue text_for_reranker selon ton standard existant.
    """
    heading = chunk_row.get('chunk_heading_full', '')
    text_original = chunk_row.get('chunk_text', '')
    visual_summary = chunk_row.get('chunk_visual_summary') or ""
    
    # Reconstitue text_for_reranker (format existant pour le reranker)
    rerank_parts = []
    if visual_summary:
        rerank_parts.append(f"[CONTENU VISUEL ET TABLEAUX]: {visual_summary}")
    if heading and heading != "Sans titre":
        rerank_parts.append(f"[TITRE/CONTEXTE]: {heading}")
    rerank_parts.append(f"[TEXTE BRUT]: {text_original}")
    
    return {
        "chunk_id": str(chunk_row.get('chunk_id', '')),
        "text_for_reranker": "\n\n".join(rerank_parts),
        "heading_full": heading,
        "page_numbers": chunk_row.get('chunk_page_numbers', []),
        "visual_summary": visual_summary,
        "tables": chunk_row.get('chunk_tables', []),
        "images_urls": chunk_row.get('chunk_images_urls', []),
        "relevance_score": chunk_row.get('relevance_score', 1.0),
        "is_identity": False
    }



# Dans strategies.py ou helpers.py

async def calculate_tag_trust_score(tag_id: uuid.UUID, conn) -> Dict:
    """
    Calcule le score de confiance d'un tag.
    
    Returns:
        {
            "trust_score": float,  # 0.0-1.0
            "reasoning": str,
            "metrics": {...}
        }
    """
    
    # ═══════════════════════════════════════════════════════════
    # Metric 1 : Nombre d'entités liées
    # ═══════════════════════════════════════════════════════════
    entities = await conn.fetch("""
        SELECT entity_id
        FROM entity_tags
        WHERE tag_id = $1
    """, tag_id)
    
    nb_entities = len(entities)
    
    # Score : 0 si <3, 0.5 si 5, 1.0 si 10+
    entity_score = min(1.0, nb_entities / 10)
    
    # ═══════════════════════════════════════════════════════════
    # Metric 2 : Coverage docs (entités du tag présentes dans combien de docs ?)
    # ═══════════════════════════════════════════════════════════
    if nb_entities > 0:
        entity_ids = [e['entity_id'] for e in entities]
        
        docs = await conn.fetch("""
            SELECT DISTINCT c.doc_id
            FROM entity_links el
            JOIN chunks c ON el.chunk_id = c.chunk_id
            WHERE el.entity_id = ANY($1::uuid[])
        """, entity_ids)
        
        nb_docs = len(docs)
        
        # Score : 0.5 si 1 doc, 1.0 si 3+ docs
        doc_coverage_score = min(1.0, nb_docs / 3)
    else:
        doc_coverage_score = 0.0
    
    # ═══════════════════════════════════════════════════════════
    # Metric 3 : Tag type (système = boost)
    # ═══════════════════════════════════════════════════════════
    tag_info = await conn.fetchrow("""
        SELECT is_system, tag_type
        FROM tags
        WHERE tag_id = $1
    """, tag_id)
    
    system_boost = 0.3 if tag_info['is_system'] else 0.0
    
    # ═══════════════════════════════════════════════════════════
    # Metric 4 (optionnel) : Cohérence sémantique des entités
    # ═══════════════════════════════════════════════════════════
    # Si toutes les entités du tag ont des types cohérents
    # Exemple : Tag "Mères" → toutes PERSONNE ✅
    #          Tag random → mix PERSONNE/LIEU/CONCEPT ❌
    
    entity_types = await conn.fetch("""
        SELECT DISTINCT e.entity_type
        FROM entities e
        JOIN entity_tags et ON e.entity_id = et.entity_id
        WHERE et.tag_id = $1
    """, tag_id)
    
    # Score : 1.0 si 1 seul type, 0.5 si 2 types, 0.0 si 3+
    nb_types = len(entity_types)
    coherence_score = max(0.0, 1.0 - (nb_types - 1) * 0.5)
    
    # ═══════════════════════════════════════════════════════════
    # Trust score final (moyenne pondérée)
    # ═══════════════════════════════════════════════════════════
    trust_score = (
        entity_score * 0.3 +          # 30% poids
        doc_coverage_score * 0.3 +    # 30% poids
        coherence_score * 0.2 +       # 20% poids
        system_boost                   # +30% si système
    )
    
    reasoning = f"""
    Tag trust analysis:
    - {nb_entities} entities linked (score: {entity_score:.2f})
    - Present in {nb_docs} document(s) (score: {doc_coverage_score:.2f})
    - {nb_types} entity type(s) (coherence: {coherence_score:.2f})
    - System tag: {tag_info['is_system']} (boost: {system_boost:.2f})
    → Trust score: {trust_score:.2f}
    """
    
    return {
        "trust_score": trust_score,
        "reasoning": reasoning,
        "metrics": {
            "nb_entities": nb_entities,
            "nb_docs": nb_docs,
            "nb_entity_types": nb_types,
            "is_system": tag_info['is_system']
        }
    }

# ═══════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════

class RetrievalStrategy(ABC):
    """Base class pour strategies de retrieval."""
    
    @abstractmethod
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        """
        Récupère les chunks pertinents.
        
        Args:
            query_data: {
                "question": str,
                "query_type": str,
                "entities": List[Dict],  # Résultats de resolve_entities_in_query
                "standalone_query": str
            }
        
        Returns:
            List[Dict] avec structure compatible retrieve_chunks :
            {
                "chunk_id": str,
                "text_for_reranker": str,
                "heading_full": str,
                "page_numbers": list,
                "visual_summary": str,
                "tables": list,
                "images_urls": list,
                "relevance_score": float,
                "source": "entity_summary|entity_chunks|cooccurrence|vector",
                "is_identity": False
            }
        """
        pass


# ═══════════════════════════════════════════════════════════
# STRATEGY 1 : EntitySummaryStrategy
# ═══════════════════════════════════════════════════════════

class EntitySummaryStrategy(RetrievalStrategy):
    """
    Utilise global_summary pour réponse rapide.
    Optimal pour entity_overview avec 1 entité.
    """
    
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        entities = query_data.get("entities", [])
        
        if not entities or entities[0].get("type") != "entity":
            return []
        
        entity = entities[0]
        entity_id = entity["entity_id"]
        
        conn = await get_connection()
        try:
            result = await conn.fetchrow("""
                SELECT global_summary, name, chunk_count
                FROM entities
                WHERE entity_id = $1
            """, uuid.UUID(entity_id))
            
            if not result or not result['global_summary']:
                return []
            
            # Chunk virtuel avec format compatible
            return [{
                "chunk_id": f"summary_{entity_id}",
                "text_for_reranker": f"[RÉSUMÉ ENTITÉ]: {result['global_summary']}",
                "heading_full": f"Résumé : {result['name']}",
                "page_numbers": [],
                "visual_summary": "",
                "tables": [],
                "images_urls": [],
                "relevance_score": 0.95,
                "source": "entity_summary",
                "is_identity": False,
                "entity_name": result['name'],
                "chunk_count": result['chunk_count']
            }]
            
        finally:
            await release_connection(conn)


# ═══════════════════════════════════════════════════════════
# STRATEGY 2 : RelationshipStrategy
# ═══════════════════════════════════════════════════════════

class RelationshipStrategy(RetrievalStrategy):
    """
    Exploite entity_cooccurrences pour trouver shared_chunks.
    Optimal pour relationship avec 2+ entités.
    """
    
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        entities = query_data.get("entities", [])
        
        if len(entities) < 2:
            return []
        
        # Prend les 2 premières entités
        entity_a_id = entities[0]["entity_id"]
        entity_b_id = entities[1]["entity_id"]
        
        conn = await get_connection()
        try:
            # Cherche la co-occurrence
            cooccurrence = await conn.fetchrow("""
                SELECT shared_chunks, co_occurrence_count
                FROM entity_cooccurrences
                WHERE (entity_a_id = $1 AND entity_b_id = $2)
                   OR (entity_a_id = $2 AND entity_b_id = $1)
            """, uuid.UUID(entity_a_id), uuid.UUID(entity_b_id))
            
            if not cooccurrence or not cooccurrence['shared_chunks']:
                print(f"   ⚠️ Aucune co-occurrence trouvée entre les 2 entités")
                return []
            
            shared_chunk_ids = cooccurrence['shared_chunks']
            print(f"   ✅ {len(shared_chunk_ids)} chunks partagés trouvés")
            
            # Récupère les chunks partagés
            chunks = await conn.fetch("""
                SELECT 
                    chunk_id,
                    chunk_text,
                    chunk_heading_full,
                    chunk_page_numbers,
                    chunk_visual_summary,
                    chunk_tables,
                    chunk_images_urls,
                    chunk_index
                FROM chunks
                WHERE chunk_id = ANY($1::uuid[])
                ORDER BY chunk_index
            """, shared_chunk_ids)
            
            # Format avec helper function
            return [
                {
                    **format_chunk_for_context(chunk),
                    "source": "cooccurrence"
                }
                for chunk in chunks
            ]
            
        finally:
            await release_connection(conn)


# ═══════════════════════════════════════════════════════════
# STRATEGY 3 : HybridStrategy
# ═══════════════════════════════════════════════════════════

class HybridStrategy(RetrievalStrategy):
    """
    Combine entity chunks + vector search.
    Optimal pour entity_overview avec entités + besoin de complétude.
    """
    
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        entities = query_data.get("entities", [])
        
        if not entities:
            return []
        
        conn = await get_connection()
        try:
            entity_chunks = []
            
            # Pour chaque entité, récupère ses chunks
            for entity in entities:
                if entity.get("type") != "entity":
                    continue
                
                entity_id = entity["entity_id"]
                entity_name = entity.get("name", "")
                
                chunks = await conn.fetch("""
                    SELECT 
                        c.chunk_id,
                        c.chunk_text,
                        c.chunk_heading_full,
                        c.chunk_page_numbers,
                        c.chunk_visual_summary,
                        c.chunk_tables,
                        c.chunk_images_urls,
                        c.chunk_index,
                        el.relevance_score
                    FROM entity_links el
                    JOIN chunks c ON el.chunk_id = c.chunk_id
                    WHERE el.entity_id = $1
                    ORDER BY el.relevance_score DESC
                    LIMIT 20
                """, uuid.UUID(entity_id))
                
                print(f"   📦 {len(chunks)} chunks trouvés pour '{entity_name}'")
                
                for chunk in chunks:
                    formatted = format_chunk_for_context(chunk)
                    formatted["source"] = "entity_chunks"
                    entity_chunks.append(formatted)
            
            return entity_chunks
            
        finally:
            await release_connection(conn)


# ═══════════════════════════════════════════════════════════
# STRATEGY 4 : VectorOnlyStrategy
# ═══════════════════════════════════════════════════════════

class VectorOnlyStrategy(RetrievalStrategy):
    """
    Fallback : utilise retrieve_chunks existant (Qdrant + reranking).
    """
    
    def __init__(self, retrieve_chunks_func):
        self.retrieve_chunks = retrieve_chunks_func
    
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        # Appelle ta fonction existante
        results = await self.retrieve_chunks(
            query_data,
            rerank_limit=20,
            limit=50
        )
        
        # Ajoute juste la source
        for chunk in results:
            chunk["source"] = "vector"
        
        return results




# ═══════════════════════════════════════════════════════════
# STRATEGY 5 : TagGroupStrategy
# ═══════════════════════════════════════════════════════════





class TagGroupStrategy(RetrievalStrategy):
    """
    Retrieval pour questions de type liste/groupe.
    Utilise metadata tag si trust score > 0.6.
    """

    async def retrieve(self, query_data: Dict) -> List[Dict]:
        entities = query_data.get("entities", [])
        
        # Vérifie type tag_group
        if not entities or entities[0].get("type") != "tag_group":
            return []
        
        tag_group = entities[0]
        tag_id = uuid.UUID(tag_group['tag_id'])  
        
        conn = await get_connection()
        try:
            # ═══════════════════════════════════════════════════════════
            # Calcule trust score
            # ═══════════════════════════════════════════════════════════
            trust_analysis = await calculate_tag_trust_score(tag_id, conn)
            trust_score = trust_analysis['trust_score']
            
            print(f"   🎯 Tag '{tag_group['tag_name']}' trust score: {trust_score:.2f}")
            # print(trust_analysis['reasoning'])  # Optionnel : verbose
            
            # Décision basée sur trust
            if trust_score < 0.6:
                print(f"   ⚠️ Trust score trop faible ({trust_score:.2f}), skip metadata")
                return []
            
            # ═══════════════════════════════════════════════════════════
            # Récupère infos tag pour description
            # ═══════════════════════════════════════════════════════════
            tag_info = await conn.fetchrow("""
                SELECT description, is_system, tag_type
                FROM tags
                WHERE tag_id = $1
            """, tag_id)
            
            entity_list = tag_group["entities"]
            
            # ═══════════════════════════════════════════════════════════
            # Metadata block : Liste COMPLÈTE
            # ═══════════════════════════════════════════════════════════
            
            entities_text = "\n".join([
                f"{i+1}. {e['name']} ({e['chunk_count']} extraits disponibles)"
                for i, e in enumerate(entity_list)
            ])
            
            metadata_block = {
                "chunk_id": f"metadata_tag_{str(tag_id)}",
                "text_for_reranker": f"""[TAXONOMIE STRUCTURÉE]

    Catégorie : {tag_group['tag_name']}
    Description : {tag_info['description'] or "Groupe d'entités reliées"}
    Fiabilité : {trust_score:.0%} (basé sur {trust_analysis['metrics']['nb_entities']} entités, {trust_analysis['metrics']['nb_docs']} document(s))

    === LISTE EXHAUSTIVE ({len(entity_list)} éléments) ===

    {entities_text}

    ─────────────────────────────────────────────────────
    Note méthodologique :
    Cette liste provient de la base de connaissances structurée.
    Les extraits de documents ci-dessous apportent le contexte narratif et les détails de chaque élément.
    Pour une réponse complète, combine cette liste avec les informations textuelles fournies.
    ─────────────────────────────────────────────────────""",
                "source": "tag_metadata",
                "relevance_score": trust_score,
                "is_identity": False,
                "heading_full": f"Taxonomie : {tag_group['tag_name']}",
                "page_numbers": [],
                "visual_summary": "",
                "tables": [],
                "images_urls": []
            }
            
            # ═══════════════════════════════════════════════════════════
            # Chunks contextuels (2 par entité, top 5 entités)
            # ═══════════════════════════════════════════════════════════
            context_chunks = []
            
            for entity in entity_list[:min(5, len(entity_list))]:
                chunks = await conn.fetch("""
                    SELECT 
                        c.chunk_id,
                        c.chunk_text,
                        c.chunk_heading_full,
                        c.chunk_page_numbers,
                        c.chunk_visual_summary,
                        c.chunk_tables,
                        c.chunk_images_urls,
                        el.relevance_score
                    FROM entity_links el
                    JOIN chunks c ON el.chunk_id = c.chunk_id
                    WHERE el.entity_id = $1
                    ORDER BY el.relevance_score DESC
                    LIMIT 2
                """, uuid.UUID(entity['entity_id']))
                
                for chunk in chunks:
                    formatted = format_chunk_for_context(chunk)
                    formatted["source"] = "tag_context"
                    context_chunks.append(formatted)
            
            print(f"   📋 Tag metadata : {len(entity_list)} entités")
            print(f"   📄 Context chunks : {len(context_chunks)} extraits")
            
            # Metadata FIRST, puis chunks
            return [metadata_block] + context_chunks
            
        finally:
            await release_connection(conn)

# ═══════════════════════════════════════════════════════════
# STRATEGY SELECTOR
# ═══════════════════════════════════════════════════════════

def select_strategy(query_type: str, entities: List[Dict], retrieve_chunks_func) -> RetrievalStrategy:
    """
    Sélectionne la stratégie optimale selon le contexte.
    
    Args:
        query_type: Type de question (entity_overview, relationship, etc.)
        entities: Liste des entités résolues
        retrieve_chunks_func: Fonction retrieve_chunks existante (pour fallback)
    
    Returns:
        Instance de RetrievalStrategy
    """
     # Pas d'entités → Vector only
    if not entities:
        print(f"   🎯 Stratégie : VectorOnlyStrategy (aucune entité)")
        return VectorOnlyStrategy(retrieve_chunks_func)
    
    
     # Tag group détecté 
    if entities[0].get("type") == "tag_group":
        print(f"   🎯 Stratégie : TagGroupStrategy (tag détecté)")
        return TagGroupStrategy()
    
   
    # 1 entité + entity_overview → Summary si disponible
    if query_type == "entity_overview" and len(entities) == 1:
        print(f"   🎯 Stratégie : EntitySummaryStrategy (1 entité)")
        return EntitySummaryStrategy()
    
    # 2+ entités + relationship → Cooccurrence
    if query_type == "relationship" and len(entities) >= 2:
        print(f"   🎯 Stratégie : RelationshipStrategy (2+ entités)")
        return RelationshipStrategy()
    
    # Autres cas avec entités → Hybrid
    if entities:
        print(f"   🎯 Stratégie : HybridStrategy ({len(entities)} entité(s))")
        return HybridStrategy()
    
    # Fallback
    print(f"   🎯 Stratégie : VectorOnlyStrategy (fallback)")
    return VectorOnlyStrategy(retrieve_chunks_func)