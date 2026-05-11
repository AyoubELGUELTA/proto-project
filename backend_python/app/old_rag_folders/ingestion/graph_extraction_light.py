import asyncio
import logging
import json
from typing import List, Dict, Any
from app.services.llm_service import LLMService

from app.ingestion.graph_prompts import get_graph_prompt 
from app.models.graph_schemas import EntityType, RelationType
from app.utils.text_utils import normalize_entity_name

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
            # --- ÉTAPE 1 : ENTITÉS ---
            sys_ent, usr_ent_template = await get_prepared_prompts("sira", "entities_p1")
            user_ent = usr_ent_template.format(
                chunk_text=chunk_text,
                identity_context=identity_context
            )
            
            entity_res = await llm_service.generate_json(sys_ent, user_ent)
            raw_entities = entity_res.get("entities", [])

            # Injection de l'ID normalisé 
            entities = []
            for e in raw_entities:
                if e.get("name"):
                    e["normalized_name"] = normalize_entity_name(e["name"])
                    entities.append(e)

            # --- ÉTAPE 2 : RELATIONS ---
            if not entities:
                return {"chunk_id": chunk_id, "entities": [], "relations": []}

            # On aide le LLM avec les noms qu'on vient d'extraire
            entity_names = [e["name"] for e in entities]
            
            sys_rel, usr_rel_template = await get_prepared_prompts("sira", "relations_p1")
            user_rel = usr_rel_template.format(
                chunk_text=chunk_text,
                entity_names=json.dumps(entity_names, ensure_ascii=False),
                identity_context=identity_context
            )
            
            relation_res = await llm_service.generate_json(sys_rel, user_rel)
            raw_relations = relation_res.get("relations", [])

            # Préparation du terrain pour le futur REWIRE, 
            relations = []
            for r in raw_relations:
                r["source_id"] = normalize_entity_name(r.get("source", ""))
                r["target_id"] = normalize_entity_name(r.get("target", ""))
                relations.append(r)

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



async def get_prepared_prompts(domain: str, step: str):
    sys_prompt, usr_template = get_graph_prompt(domain, step)
    
    # On remplace les placeholders de taxonomie une fois pour toutes
    if "entities" in step:
        sys_prompt = sys_prompt.replace("{entity_taxonomy}", EntityType.get_llm_definitions())
    elif "relations" in step:
        sys_prompt = sys_prompt.replace("{relation_taxonomy}", RelationType.get_llm_definitions())
        
    return sys_prompt, usr_template