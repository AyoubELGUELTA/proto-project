"""
Post-processing Graph RAG - Deduplication par Clustering + LLM ciblé
"""

from typing import List, Dict, Tuple, Any
from app.utils.text_utils import normalize_entity_name
from app.logic import graph_cleaner as gc



# ============================================================
# PIPELINE CLUSTERING COMPLET
# ============================================================

async def deduplicate_entities_clustered(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Deduplication par clustering + LLM ciblé
    
    Workflow:
    1. Build similarity graph (threshold 0.5)
    2. Find connected components (clusters)
    3. LLM arbitre chaque cluster > 1 entity
    4. Apply merges
    5. Update relations
    
    Returns: (deduplicated_entities, updated_relations)
    """
    
    print(f"\n🔧 Clustering Deduplication: {len(entities)} entities...")
    
    if len(entities) < 2:
        print("  ℹ️  Moins de 2 entities, skip clustering")
        return entities, relations
    
    # PHASE 1: Build similarity graph
    print("  📊 Phase 1: Build similarity graph (threshold 0.5)...")
    graph = gc.build_similarity_graph(entities, threshold=0.5)
    
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"     → {edge_count} edges détectées")
    
    # PHASE 2: Find clusters
    print("  🔍 Phase 2: Find connected components...")
    clusters = gc.find_connected_components(graph, len(entities))
    
    # Séparer singletons vs multi-entity clusters
    singleton_clusters = [c for c in clusters if len(c) == 1]
    multi_clusters = [c for c in clusters if len(c) > 1]
    
    print(f"     → {len(multi_clusters)} clusters suspects (multi-entity)")
    print(f"     → {len(singleton_clusters)} singletons (OK)")
    
    if not multi_clusters:
        print("  ✅ Aucun cluster suspect, pas de deduplication nécessaire\n")
        return entities, relations
    
    # PHASE 3: LLM arbitrage
    print(f"  🤖 Phase 3: LLM arbitrage {len(multi_clusters)} clusters...\n")
    
    llm_decisions = []
    
    for idx, cluster_indices in enumerate(multi_clusters, start=1):
        cluster_entities = [entities[i] for i in cluster_indices]
        
        print(f"     Cluster #{idx}: {len(cluster_entities)} entities")
        
        if len(cluster_entities) > 15:
            print(f"       ⚠️  Large cluster ({len(cluster_entities)} entities) - using gpt-4o")
        
        decision = await gc.arbitrate_cluster_llm(cluster_entities, idx)
        llm_decisions.append(decision)
        
        # Log actions
        for action in decision.get("actions", []):
            if action["type"] == "merge":
                print(f"       🔗 Merge: {len(action.get('entity_ids', []))} entities → '{action.get('canonical_name')}'")
            elif action["type"] == "keep_distinct":
                print(f"       ✅ Keep distinct: {len(action.get('entity_ids', []))} entities")
    
    # PHASE 4: Apply decisions
    print("\n  🔨 Phase 4: Apply merges...")
    # entities_dedup: liste d'objets entités fusionnés
    # name_mapping: Dict[old_id, new_id]

    entities_dedup, name_mapping = gc.apply_cluster_decisions(entities, multi_clusters, llm_decisions)
    
    # PHASE 5: Update relations (Rewiring)
    # On utilise ENFIN la fonction dédiée de graph_cleaner
    print("  🔗 Phase 5: Update relations (Rewiring)...")
    relations_updated = gc.rewire_relations(relations, name_mapping)
    
    return entities_dedup, relations_updated


# ============================================================
# PIPELINE COMPLÈTE
# ============================================================

async def deduplicate_entities_clustered(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]]
) -> Tuple[List[Dict], List[Dict]]:
    # ... (Phases 1 à 3 identiques) ...

    # PHASE 4: Apply decisions
    print("\n  🔨 Phase 4: Apply merges...")
    # entities_dedup: liste d'objets entités fusionnés
    # name_mapping: Dict[old_id, new_id]
    entities_dedup, name_mapping = gc.apply_cluster_decisions(entities, multi_clusters, llm_decisions)
    
    # PHASE 5: Update relations (Rewiring)
    # On utilise ENFIN la fonction dédiée de graph_cleaner
    print("  🔗 Phase 5: Update relations (Rewiring)...")
    relations_updated = gc.rewire_relations(relations, name_mapping)
    
    return entities_dedup, relations_updated


async def post_process_graph_extraction(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Pipeline post-processing complète"""
    
    # 1. Deduplication (inclut maintenant le rewire automatique)
    entities_dedup, relations_updated = await deduplicate_entities_clustered(entities, relations)

    # 2. Alias cleanup
    entities_clean = gc.cleanup_common_aliases(entities_dedup)
    
    # 3. Validation théologique
    validation = await gc.validate_theology(entities_clean) # Ne fait pas grand chose pour l'instant, si ce n'est quelques warnings, a réfléchir plus tard TODO
    
    # 4. Clean orphaned relations (CORRIGÉ)
    # On utilise les IDs NORMALISÉS pour la vérification, c'est infaillible
    valid_entity_ids = {e["normalized_name"] for e in entities_clean}
    
    relations_clean = [
        rel for rel in relations_updated
        if (rel.get("source_id") in valid_entity_ids and
            rel.get("target_id") in valid_entity_ids)
    ]
    
    orphan_count = len(relations_updated) - len(relations_clean)
    if orphan_count > 0:
        print(f"⚠️  {orphan_count} relations orphelines supprimées")
    
    return entities_clean, relations_clean, validation

async def prepare_for_post_processing(raw_entities, raw_relations):
    # 1. On normalise les entités et on leur donne un ID technique
    for ent in raw_entities:
        ent['normalized_name'] = normalize_entity_name(ent.get('name', ''))    

    # 2. On prépare les relations pour le matching futur

    for rel in raw_relations:
        # On normalise ce que le LLM a écrit dans source/target
        rel['source_id'] = normalize_entity_name(rel.get('source', ''))
        rel['target_id'] = normalize_entity_name(rel.get('target', ''))
        
    return raw_entities, raw_relations
        
