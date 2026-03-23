"""
Graph RAG Extraction - Pipeline complète
Entities + Relations extraction avec 2 passes LLM
"""

import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI

from app.core.config import OPENAI_API_KEY
from app.ingestion.graph_prompts import get_prompts
from app.ingestion.graph_schemas import EntityType, RelationType

# ============================================================
# CONFIGURATION MODÈLES
# ============================================================

# Entities: gpt-4o-mini (bon compromis coût/qualité)
ENTITY_MODEL = "gpt-4o-mini-2024-07-18"

# Relations: gpt-4o (meilleur compréhension contexte)
RELATION_MODEL = "gpt-4o-mini-2024-07-18"

# Concurrence
SEMAPHORE = asyncio.Semaphore(5)  # 5 chunks parallèles max

# ============================================================
# EXTRACTION ENTITIES - PASS 1
# ============================================================

async def extract_entities_pass1(
    chunk_text: str,
    identity_context: str,
    domain: str = "sira"
) -> List[Dict[str, Any]]:
    """
    Extraction initiale entities depuis chunk
    
    Args:
        chunk_text: Texte du chunk
        identity_context: Fiche identité document
        domain: "sira" (extensible: "fiqh", "hadith")
    
    Returns:
        Liste entities extraites
    """
    prompts = get_prompts(domain)
    
    prompt = prompts["entity_pass1"].format(
        identity_context=identity_context,
        chunk_text=chunk_text
    )
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=ENTITY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Déterministe
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        entities = result.get("entities", [])
        
        print(f"  ✅ Pass 1: {len(entities)} entities extraites")
        return entities
    
    except Exception as e:
        print(f"  ❌ Erreur extraction entities Pass 1: {e}")
        return []

# ============================================================
# EXTRACTION ENTITIES - PASS 2 (GLEANING)
# ============================================================

async def extract_entities_pass2(
    chunk_text: str,
    identity_context: str,
    entities_pass1: List[Dict[str, Any]],
    domain: str = "sira"
) -> List[Dict[str, Any]]:
    """
    Review entities Pass 1 + extraction entities ratées
    
    Args:
        chunk_text: Texte du chunk
        identity_context: Fiche identité
        entities_pass1: Entities extraites Pass 1
        domain: "sira"
    
    Returns:
        Entities additionnelles trouvées
    """
    prompts = get_prompts(domain)
    
    prompt = prompts["entity_pass2"].format(
        identity_context=identity_context,
        chunk_text=chunk_text,
        entities_pass1=json.dumps(entities_pass1, ensure_ascii=False, indent=2)
    )
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=ENTITY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        additional = result.get("additional_entities", [])
        
        print(f"  ✅ Pass 2: {len(additional)} entities additionnelles")
        return additional
    
    except Exception as e:
        print(f"  ❌ Erreur extraction entities Pass 2: {e}")
        return []

# ============================================================
# EXTRACTION RELATIONS - PASS 1
# ============================================================

async def extract_relations_pass1(
    chunk_text: str,
    identity_context: str,
    final_entities: List[Dict[str, Any]],
    domain: str = "sira"
) -> List[Dict[str, Any]]:
    """
    Extraction relations entre entities validées
    
    Args:
        chunk_text: Texte du chunk
        identity_context: Fiche identité
        final_entities: Entities complètes (Pass 1 + Pass 2)
        domain: "sira"
    
    Returns:
        Liste relations extraites
    """
    prompts = get_prompts(domain)
    
    prompt = prompts["relation_pass1"].format(
        identity_context=identity_context,
        chunk_text=chunk_text,
        final_entities=json.dumps(final_entities, ensure_ascii=False, indent=2)
    )
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=RELATION_MODEL,  # GPT-4o pour meilleure compréhension
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        relations = result.get("relations", [])
        
        print(f"  ✅ Relations Pass 1: {len(relations)} extraites")
        return relations
    
    except Exception as e:
        print(f"  ❌ Erreur extraction relations Pass 1: {e}")
        return []

# ============================================================
# EXTRACTION RELATIONS - PASS 2 (GLEANING)
# ============================================================

async def extract_relations_pass2(
    chunk_text: str,
    identity_context: str,
    final_entities: List[Dict[str, Any]],
    relations_pass1: List[Dict[str, Any]],
    domain: str = "sira"
) -> List[Dict[str, Any]]:
    """
    Review relations Pass 1 + extraction relations ratées
    
    Args:
        chunk_text: Texte du chunk
        identity_context: Fiche identité
        final_entities: Entities validées
        relations_pass1: Relations extraites Pass 1
        domain: "sira"
    
    Returns:
        Relations additionnelles trouvées
    """
    prompts = get_prompts(domain)
    
    prompt = prompts["relation_pass2"].format(
        identity_context=identity_context,
        chunk_text=chunk_text,
        final_entities=json.dumps(final_entities, ensure_ascii=False, indent=2),
        relations_pass1=json.dumps(relations_pass1, ensure_ascii=False, indent=2)
    )
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=RELATION_MODEL,  # GPT-4o pour review critique
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        additional = result.get("additional_relations", [])
        
        print(f"  ✅ Relations Pass 2: {len(additional)} additionnelles")
        return additional
    
    except Exception as e:
        print(f"  ❌ Erreur extraction relations Pass 2: {e}")
        return []

# ============================================================
# FONCTION MAÎTRE - PROCESS SINGLE CHUNK
# ============================================================

async def process_chunk_graph_extraction(
    chunk_data: Dict[str, Any],
    identity_context: str,
    domain: str = "sira"
) -> Dict[str, Any]:
    """
    Pipeline complète extraction Graph pour 1 chunk
    
    Workflow:
    1. Entities Pass 1 (extraction initiale)
    2. Entities Pass 2 (gleaning)
    3. Merge entities
    4. Relations Pass 1 (extraction initiale)
    5. Relations Pass 2 (gleaning)
    6. Merge relations
    
    Args:
        chunk_data: {text, heading_full, chunk_id, ...}
        identity_context: Fiche identité document
        domain: "sira"
    
    Returns:
        {
            chunk_id,
            entities: [...],
            relations: [...],
            stats: {entities_pass1, entities_pass2, relations_pass1, relations_pass2}
        }
    """
    chunk_text = chunk_data.get("text", "")
    chunk_id = chunk_data.get("chunk_id")
    
    print(f"\n🔄 Processing chunk {chunk_id}...")
    
    # PASS 1: Entities
    entities_p1 = await extract_entities_pass1(chunk_text, identity_context, domain)
    
    # PASS 2: Entities gleaning
    entities_p2 = await extract_entities_pass2(
        chunk_text, identity_context, entities_p1, domain
    )
    
    # Merge entities
    final_entities = entities_p1 + entities_p2
    
    # PASS 1: Relations
    relations_p1 = await extract_relations_pass1(
        chunk_text, identity_context, final_entities, domain
    )
    
    # PASS 2: Relations gleaning
    relations_p2 = await extract_relations_pass2(
        chunk_text, identity_context, final_entities, relations_p1, domain
    )
    
    # Merge relations
    final_relations = relations_p1 + relations_p2
    
    print(f"✅ Chunk {chunk_id} terminé:")
    print(f"   Entities: {len(entities_p1)} + {len(entities_p2)} = {len(final_entities)}")
    print(f"   Relations: {len(relations_p1)} + {len(relations_p2)} = {len(final_relations)}")
    
    return {
        "chunk_id": chunk_id,
        "entities": final_entities,
        "relations": final_relations,
        "stats": {
            "entities_pass1": len(entities_p1),
            "entities_pass2": len(entities_p2),
            "relations_pass1": len(relations_p1),
            "relations_pass2": len(relations_p2),
        }
    }

# ============================================================
# FONCTION MAÎTRE - PROCESS MULTIPLE CHUNKS
# ============================================================

async def extract_graph_from_chunks(
    chunks: List[Dict[str, Any]],
    identity_context: str,
    domain: str = "sira"
) -> Dict[str, Any]:
    """
    Extraction Graph pour tous les chunks d'un document
    
    Args:
        chunks: Liste chunks [{text, chunk_id, ...}, ...]
        identity_context: Fiche identité document
        domain: "sira"
    
    Returns:
        {
            entities: [...],  # Toutes entities tous chunks
            relations: [...],  # Toutes relations
            chunks_processed: int,
            total_stats: {...}
        }
    """
    print(f"\n{'='*70}")
    print(f"🚀 GRAPH EXTRACTION - {len(chunks)} chunks")
    print(f"{'='*70}\n")
    
    # Process chunks avec concurrence limitée
    async def bounded_process(chunk):
        async with SEMAPHORE:
            return await process_chunk_graph_extraction(chunk, identity_context, domain)
    
    tasks = [bounded_process(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    all_entities = []
    all_relations = []
    total_stats = {
        "entities_pass1": 0,
        "entities_pass2": 0,
        "relations_pass1": 0,
        "relations_pass2": 0,
    }
    
    for result in results:
        all_entities.extend(result["entities"])
        all_relations.extend(result["relations"])
        
        for key in total_stats:
            total_stats[key] += result["stats"][key]
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION TERMINÉE")
    print(f"{'='*70}")
    print(f"Chunks traités: {len(chunks)}")
    print(f"Entities totales: {len(all_entities)}")
    print(f"  - Pass 1: {total_stats['entities_pass1']}")
    print(f"  - Pass 2: {total_stats['entities_pass2']}")
    print(f"Relations totales: {len(all_relations)}")
    print(f"  - Pass 1: {total_stats['relations_pass1']}")
    print(f"  - Pass 2: {total_stats['relations_pass2']}")
    print(f"{'='*70}\n")
    
    return {
        "entities": all_entities,
        "relations": all_relations,
        "chunks_processed": len(chunks),
        "total_stats": total_stats
    }