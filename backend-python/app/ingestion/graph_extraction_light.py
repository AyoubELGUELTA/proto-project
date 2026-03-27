import asyncio
import logging
import json
from typing import List, Dict, Any
from app.services.llm_service import LLMService

from app.ingestion.graph_prompts import get_graph_prompt 

logger = logging.getLogger(__name__)
SEMAPHORE = asyncio.Semaphore(15) 

async def process_chunk_light(
    chunk_data: Dict[str, Any],
    llm_service: LLMService,
    identity_context: str
) -> Dict[str, Any]:
    chunk_text = chunk_data.get("text", "")
    chunk_id = chunk_data.get("chunk_id")
    
    async with SEMAPHORE:
        try:
            # --- ÉTAPE 1 : ENTITÉS (P1) ---
            sys_ent, usr_ent_template = get_graph_prompt("sira", "entities_p1")
            user_ent = usr_ent_template.format(
                chunk_text=chunk_text,
                identity_context=identity_context
            )
            
            # Note: il faut que LLMService accepte (system, user)
            entity_res = await llm_service.generate_json(
                system_prompt=sys_ent, 
                user_prompt=user_ent
            )
            entities = entity_res.get("entities", [])

            # --- ÉTAPE 2 : RELATIONS (P1) ---
            if not entities:
                return {"chunk_id": chunk_id, "entities": [], "relations": []}

            entity_names = [e.get("name") for e in entities]
            
            sys_rel, usr_rel_template = get_graph_prompt("sira", "relations_p1")
            user_rel = usr_rel_template.format(
                chunk_text=chunk_text,
                entity_names=json.dumps(entity_names, ensure_ascii=False),
                identity_context=identity_context
            )
            
            relation_res = await llm_service.generate_json(
                system_prompt=sys_rel, 
                user_prompt=user_rel
            )
            relations = relation_res.get("relations", [])

            return {
                "chunk_id": chunk_id,
                "entities": entities,
                "relations": relations
            }

        except Exception as e:
            logger.error(f"❌ Erreur chunk {chunk_id}: {str(e)}")
            return {"chunk_id": chunk_id, "entities": [], "relations": []}
async def run_light_extraction(chunks: List[Dict[str, Any]], identity_context: str):
    llm_service = LLMService() # Utilise gpt-4o-mini par défaut
    
    tasks = [process_chunk_light(c, llm_service, identity_context) for c in chunks]
    results = await asyncio.gather(*tasks)
    
    # Agrégation simple
    all_entities = []
    all_relations = []
    for r in results:
        all_entities.extend(r["entities"])
        all_relations.extend(r["relations"])
        
    return all_entities, all_relations
