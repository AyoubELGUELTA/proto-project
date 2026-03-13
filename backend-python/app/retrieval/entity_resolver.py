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
                        
                        # ✅ Logging amélioré avec match_type
                        print(f"      ✅ TROUVÉ : {result['name']} (match: {variant}, type: {result.get('match_type', 'N/A')}, score: {result.get('match_score', 0):.2f}, chunks: {result['chunk_count']})")
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
    """
    Cherche une entité avec scoring pondéré.
    
    Priorités :
    1. Exact match normalized_name (score 1.0)
    2. Substring match normalized_name (score 0.8)
    3. Alias match normalized_aliases (score 0.6)
    """
    
    normalized = normalize_entity_name(search_term)
    
    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 1 : Exact match normalized_name (priorité absolue)
    # ═══════════════════════════════════════════════════════════
    entity = await conn.fetchrow("""
        SELECT entity_id, name, entity_type, chunk_count, aliases, normalized_name
        FROM entities
        WHERE normalized_name = $1
    """, normalized)
    
    if entity:
        return {
            "type": "entity",
            "entity_id": str(entity['entity_id']),
            "name": entity['name'],
            "entity_type": entity['entity_type'],
            "chunk_count": entity['chunk_count'],
            "match_type": "exact",
            "match_score": 1.0
        }
    
    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 2 : Substring match normalized_name
    # ═══════════════════════════════════════════════════════════
    # "khadija" in "khadija bint khuwaylid" 
    substring_matches = await conn.fetch("""
        SELECT entity_id, name, entity_type, chunk_count, aliases, normalized_name
        FROM entities
        WHERE normalized_name LIKE '%' || $1 || '%'
           OR $1 LIKE '%' || normalized_name || '%'
    """, normalized)
    
    # ═══════════════════════════════════════════════════════════
    # ÉTAPE 3 : Alias match normalized_aliases
    # ═══════════════════════════════════════════════════════════
    alias_matches = await conn.fetch("""
        SELECT entity_id, name, entity_type, chunk_count, aliases, normalized_name, normalized_aliases
        FROM entities
        WHERE $1 = ANY(normalized_aliases)
    """, normalized)
    
    # ═══════════════════════════════════════════════════════════
    # FUSION & SCORING
    # ═══════════════════════════════════════════════════════════
    all_candidates = []
    
    # Ajoute substring matches (score 0.8)
    for entity in substring_matches:
        all_candidates.append({
            "entity_id": str(entity['entity_id']),
            "name": entity['name'],
            "entity_type": entity['entity_type'],
            "chunk_count": entity['chunk_count'],
            "match_type": "substring",
            "match_score": 0.8,
            "normalized_name": entity['normalized_name']
        })
    
    # Ajoute alias matches (score 0.6)
    for entity in alias_matches:
        # Vérifie si déjà dans candidates (via substring)
        entity_id = str(entity['entity_id'])
        if not any(c['entity_id'] == entity_id for c in all_candidates):
            all_candidates.append({
                "entity_id": entity_id,
                "name": entity['name'],
                "entity_type": entity['entity_type'],
                "chunk_count": entity['chunk_count'],
                "match_type": "alias",
                "match_score": 0.6,
                "normalized_name": entity['normalized_name']
            })
    
    # ═══════════════════════════════════════════════════════════
    # TRI : score DESC, puis chunk_count DESC
    # ═══════════════════════════════════════════════════════════
    if not all_candidates:
        return None
    
    # Trie par priorité : match_score d'abord, puis chunk_count
    all_candidates.sort(
        key=lambda x: (x['match_score'], x['chunk_count']),
        reverse=True
    )
    
    best_match = all_candidates[0]
    
    # Retourne le meilleur candidat
    return {
        "type": "entity",
        "entity_id": best_match['entity_id'],
        "name": best_match['name'],
        "entity_type": best_match['entity_type'],
        "chunk_count": best_match['chunk_count'],
        "match_type": best_match['match_type'],
        "match_score": best_match['match_score']
    }


async def _search_tag(conn, tag_name: str) -> Optional[Dict]:
    """
    Cherche tag avec normalization (comme entities).
    
    Stratégies :
    1. Label exact (normalized)
    2. Substring label (normalized)
    3. Normalized aliases match (GIN index, rapide)
    """
    
    normalized_search = normalize_entity_name(tag_name)
    
    # ═══════════════════════════════════════════════════════════
    # 1. Match exact sur label normalisé
    # ═══════════════════════════════════════════════════════════
    tag = await conn.fetchrow("""
        SELECT tag_id, label, aliases, is_system, normalized_aliases
        FROM tags
        WHERE LOWER(label) = $1
    """, normalized_search)
    
    if tag:
        print(f"      ✅ TAG exact match: {tag['label']}")
        return await build_tag_result(tag, conn)
    
    # ═══════════════════════════════════════════════════════════
    # 2. Substring match label normalisé
    # ═══════════════════════════════════════════════════════════
    # "meres croyants" in "meres des croyants"
    tags = await conn.fetch("""
        SELECT tag_id, label, aliases, is_system, normalized_aliases
        FROM tags
    """)
    
    for tag in tags:
        label_norm = normalize_entity_name(tag['label'])
        if normalized_search in label_norm or label_norm in normalized_search:
            print(f"      ✅ TAG substring match: {tag['label']}")
            return await build_tag_result(tag, conn)
    
    # ═══════════════════════════════════════════════════════════
    # 3. Match dans normalized_aliases (GIN index, rapide)
    # ═══════════════════════════════════════════════════════════
    tag = await conn.fetchrow("""
        SELECT tag_id, label, aliases, is_system, normalized_aliases
        FROM tags
        WHERE $1 = ANY(normalized_aliases)
    """, normalized_search)
    
    if tag:
        # Trouve quel alias a matché
        matched_alias = None
        if tag['aliases']:
            for alias in tag['aliases']:
                if normalize_entity_name(alias) == normalized_search:
                    matched_alias = alias
                    break
        
        print(f"      ✅ TAG alias match: {tag['label']} (via '{matched_alias or 'normalized'}')")
        return await build_tag_result(tag, conn)
    
    # Pas trouvé
    return None


async def build_tag_result(tag, conn):
    """Helper construction résultat tag."""
    entities = await conn.fetch("""
        SELECT e.entity_id, e.name, e.chunk_count
        FROM entities e
        JOIN entity_tags et ON e.entity_id = et.entity_id
        WHERE et.tag_id = $1
    """, tag['tag_id'])
    
    return {
        "type": "tag_group",
        "tag_name": tag['label'],
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