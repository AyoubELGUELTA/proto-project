from typing import Dict, Any, List

class RetrievalStrategy:
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        raise NotImplementedError

class EntitySummaryStrategy(RetrievalStrategy):
    """Utilise global_summary pour réponse rapide"""
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        # Récupère global_summary de l'entité
        pass

class HybridStrategy(RetrievalStrategy):
    """Entity chunks + vector search"""
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        # 1. Récupère chunks liés à l'entité
        # 2. Lance vector search en parallèle
        # 3. Fusionne + deduplicate
        pass

class RelationshipStrategy(RetrievalStrategy):
    """Exploite entity_cooccurrences"""
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        # Trouve shared_chunks entre 2 entités
        pass

class VectorOnlyStrategy(RetrievalStrategy):
    """Fallback : ton code actuel"""
    async def retrieve(self, query_data: Dict) -> List[Dict]:
        # Ton retrieve_chunks existant
        pass