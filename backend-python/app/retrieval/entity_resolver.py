from typing import List, Dict, Any, Optional
from app.db.base import get_connection, release_connection
from app.db.entities import normalize_entity_name

async def resolve_entities_in_query(entities_mentioned: List[Dict]) -> List[Dict[str, Any]]:
    """
    Résout les entités mentionnées dans la query vers des entity_id en BDD.
    Évite les doublons si plusieurs mentions pointent vers la même entité.
    """
    
    conn = await get_connection()
    resolved = []
    
    resolved_ids = set()
    resolved_tag_ids = set()
    
    try:
        print(f"🔍 ENTITY RESOLUTION : {len(entities_mentioned)} entité(s) à résoudre")
        
        for entity_mention in entities_mentioned:
            primary = entity_mention.get("primary", "")
            variants = entity_mention.get("variants", [])
            all_variants = [primary] + variants
            
            print(f"   🔎 Recherche : {primary} | Variantes : {variants}")
            
            result = None
            
            # 1. Tentative entité individuelle
            for variant in all_variants:
                result = await _search_entity(conn, variant)
                
                if result:
                    e_id = result["entity_id"]
                    if e_id not in resolved_ids:
                        match_score = _compute_match_score(variant, result["name"])
                        resolved.append({
                            **result,
                            "match_variant": variant,
                            "match_score": match_score
                        })
                        resolved_ids.add(e_id)
                        print(f"      ✅ TROUVÉ : {result['name']} (match: {variant}, score: {match_score:.2f}, chunks: {result['chunk_count']})")
                    else:
                        print(f"      ⚠️ Doublon ignoré : {result['name']} (déjà résolu)")
                    break
            
            # 2. Si pas trouvé, cherche tag
            if not result:
                tag_result = await _search_tag(conn, primary)
                if tag_result:
                    t_id = tag_result["tag_id"]
                    if t_id not in resolved_tag_ids:
                        resolved.append(tag_result)
                        resolved_tag_ids.add(t_id)
                        print(f"      ✅ TAG TROUVÉ : {tag_result['tag_name']} ({len(tag_result['entities'])} entités)")
                    else:
                        print(f"      ⚠️ Tag doublon ignoré : {tag_result['tag_name']}")
                else:
                    print(f"      ❌ NON TROUVÉ : {primary}")
        
        print(f"🎯 RÉSULTAT : {len(resolved)} entité(s) résolue(s)")
        return resolved
        
    finally:
        await release_connection(conn)

async def _search_entity(conn, search_term: str) -> Optional[Dict]:
    """Cherche via normalized_aliases (optimisé)."""
    
    normalized = normalize_entity_name(search_term)
    
    # 1. Exact match sur normalized_name
    entity = await conn.fetchrow("""
        SELECT entity_id, name, entity_type, chunk_count, aliases
        FROM entities
        WHERE normalized_name = $1
    """, normalized)
    
    if entity:
        return {
            "type": "entity",
            "entity_id": str(entity['entity_id']),
            "name": entity['name'],
            "entity_type": entity['entity_type'],
            "chunk_count": entity['chunk_count']
        }
    
    # 2. Match via normalized_aliases (index GIN)
    entities = await conn.fetch("""
        SELECT entity_id, name, entity_type, chunk_count, aliases
        FROM entities
        WHERE $1 = ANY(normalized_aliases)
    """, normalized)
    
    if entities:
        # Prend celui avec le plus de chunks
        best = max(entities, key=lambda e: e['chunk_count'])
        return {
            "type": "entity",
            "entity_id": str(best['entity_id']),
            "name": best['name'],
            "entity_type": best['entity_type'],
            "chunk_count": best['chunk_count']
        }
    
    return None
async def _search_tag(conn, tag_name: str) -> Optional[Dict]:
    """Cherche un tag système et renvoie le groupe d'entités liées."""
    tag = await conn.fetchrow("""
        SELECT tag_id, name FROM system_tags
        WHERE name ILIKE $1
    """, f"%{tag_name}%")
    
    if not tag:
        return None
    
    entities = await conn.fetch("""
        SELECT e.entity_id, e.name, e.chunk_count
        FROM entities e
        JOIN entity_tags et ON e.entity_id = et.entity_id
        WHERE et.tag_id = $1
    """, tag['tag_id'])
    
    return {
        "type": "tag_group",
        "tag_name": tag['name'],
        "tag_id": str(tag['tag_id']),
        "entities": [
            {
                "entity_id": str(e['entity_id']),
                "name": e['name'],
                "chunk_count": e['chunk_count']
            }
            for e in entities
        ]
    }

def _compute_match_score(search_term: str, found_name: str) -> float:
    """Calcule la proximité textuelle entre le terme cherché et le nom trouvé."""
    from difflib import SequenceMatcher
    
    search_norm = normalize_entity_name(search_term)
    found_norm = normalize_entity_name(found_name)
    
    return SequenceMatcher(None, search_norm, found_norm).ratio()