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