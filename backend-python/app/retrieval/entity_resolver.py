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
    
    # Sets pour suivre ce qui a déjà été ajouté au résultat final
    resolved_ids = set()
    resolved_tag_ids = set()
    
    try:
        for entity_mention in entities_mentioned:
            primary = entity_mention.get("primary", "")
            variants = entity_mention.get("variants", [])
            all_variants = [primary] + variants
            
            result = None
            
            # 1. Tentative de résolution par entité individuelle (nom ou alias)
            for variant in all_variants:
                result = await _search_entity(conn, variant)
                
                if result:
                    # On vérifie si on n'a pas déjà ajouté cette entité via une autre mention
                    e_id = result["entity_id"]
                    if e_id not in resolved_ids:
                        resolved.append({
                            **result,
                            "match_variant": variant,
                            "match_score": _compute_match_score(variant, result["name"])
                        })
                        resolved_ids.add(e_id)
                    break  # Trouvé, on passe à l'entité mentionnée suivante
            
            # 2. Si pas trouvé en entité, on cherche dans les tags système (groupes)
            if not result:
                tag_result = await _search_tag(conn, primary)
                if tag_result:
                    t_id = tag_result["tag_id"]
                    if t_id not in resolved_tag_ids:
                        resolved.append(tag_result)
                        resolved_tag_ids.add(t_id)
        
        return resolved
        
    finally:
        await release_connection(conn)

async def _search_entity(conn, search_term: str) -> Optional[Dict]:
    """Cherche une entité par nom/alias."""
    normalized = normalize_entity_name(search_term)
    
    # 1. Match exact sur le nom normalisé (Indexé, très rapide)
    entity = await conn.fetchrow("""
        SELECT entity_id, name, entity_type, chunk_count
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
    
    # 2. Match exact dans la liste des alias
    # Note: On utilise ANY pour scanner le tableau d'aliases
    entities = await conn.fetch("""
        SELECT entity_id, name, entity_type, chunk_count
        FROM entities
        WHERE $1 = ANY(aliases)
    """, search_term)

    if entities:
        # Désambiguïsation : on prend l'entité la plus "importante" (plus de chunks)
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