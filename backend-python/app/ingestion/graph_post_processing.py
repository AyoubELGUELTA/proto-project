"""
Post-processing Graph RAG - Deduplication par Clustering + LLM ciblé
"""

import json
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
from openai import AsyncOpenAI

from app.core.config import OPENAI_API_KEY
from app.db.entities import normalize_entity_name, similarity

TYPE_PRIORITY = {
    "Prophet": 100,
    "MotherBeliever": 90,
    "AhlBayt": 85,
    "Sahabi": 80,
    "Sahabiya": 75,
    "City": 60,
    "Place": 50,
    "Battle": 40,
    "Event": 30,
    "Tribe": 20,
    "Group": 10,
    "Child": 5,
    "Location": 1
}

# ============================================================
# CLUSTERING - CONNECTED COMPONENTS
# ============================================================

def calculate_alias_overlap_fuzzy(aliases1: set, aliases2: set) -> float:
    """
    Compare deux ensembles d'alias en fuzzy match.
    Seuil de correspondance : 0.8
    - 2+ correspondances -> 1.0
    - 1 seule correspondance -> 0.7
    - 0 correspondance -> 0.0
    """
    if not aliases1 or not aliases2:
        return 0.0
    
    match_count = 0
    # On transforme en liste pour pouvoir itérer proprement si besoin, 
    # mais le set est déjà bien pour l'unicité
    list1 = list(aliases1)
    list2 = list(aliases2)
    
    # On garde trace des éléments de list2 déjà "matchés" pour ne pas 
    # compter deux fois le même alias si list1 a des quasi-doublons
    matched_indices = set()

    for a1 in list1:
        for i, a2 in enumerate(list2):
            if i in matched_indices:
                continue
                
            # Utilisation de ta fonction similarity existante
            if similarity(a1, a2) >= 0.8:
                match_count += 1
                matched_indices.add(i)
                break # On a trouvé un match pour a1, on passe à l'alias suivant de list1
        
        # Optimisation : si on a déjà 2 matchs, on peut s'arrêter (seuil max atteint)
        if match_count >= 2:
            return 1.0

    if match_count == 1:
        return 0.7
        
    return 0.0

def calculate_similarity_score(e1: Dict, e2: Dict) -> float:
    """Score multi-critères entre 2 entities"""
    
    scores = []
    
    # 1. Name similarity (poids: 0.5)
    name_sim = similarity(e1["normalized_name"], e2["normalized_name"])
    scores.append(name_sim * 0.5)
    
    # 2. Alias overlap (poids: 0.3)
    aliases1 = {normalize_entity_name(a) for a in e1.get('aliases', [])}
    aliases2 = {normalize_entity_name(a) for a in e2.get('aliases', [])}
    
    # Ajouter names dans aliases pour comparison
    aliases1.add(e1["normalized_name"])
    aliases2.add(e2["normalized_name"])
    
    scores.append(calculate_alias_overlap_fuzzy(aliases1, aliases2) * 0.3)
    
    # 3. Type compatibility (poids: 0.1)
    type_compat = types_compatible(e1["type"], e2["type"])
    scores.append(type_compat * 0.1)
    
    # 4. Context similarity (poids: 0.1)
    ctx1 = e1.get("context_description", "")
    ctx2 = e2.get("context_description", "")
    
    if ctx1 and ctx2:
        ctx_sim = similarity(ctx1[:100], ctx2[:100])
    else:
        ctx_sim = 0
    
    scores.append(ctx_sim * 0.1)
    
    return sum(scores)

def types_compatible(type1: str, type2: str) -> float:
    """
    Compatibilité entre types
    
    Returns:
    - 1.0: Identique
    - 0.7: Compatible (peut être même entity)
    - 0.3: Neutre
    - 0.0: Incompatible (jamais même entity)
    """
    
    if type1 == type2:
        return 1.0
    
    # Compatible types
    compatible_sets = [
        {"AhlBayt", "Sahabi"},  # Ali peut être les deux
        {"Child", "AhlBayt"},
        {"Place", "City"},
        {"Event", "Battle"},
    ]
    
    types_set = {type1, type2}
    
    for compat_set in compatible_sets:
        if types_set.issubset(compat_set):
            return 0.7
    
    # Incompatible CRITICAL
    incompatible_sets = [
        {"MotherBeliever", "Prophet"},
        {"MotherBeliever", "AhlBayt"},  # Fatima fille ≠ Mère
        {"Sahabi", "Prophet"},
    ]
    
    for incompat_set in incompatible_sets:
        if types_set == incompat_set:
            return 0.0
    
    return 0.3


def build_similarity_graph(entities: List[Dict], threshold: float = 0.5) -> Dict[int, Set[int]]:
    """
    Construit graph de similarité
    
    Returns: Adjacency list {entity_idx: {connected_entity_idx, ...}}
    
    Example:
        {
            0: {1, 2},    # Entity 0 connectée à 1 et 2
            1: {0, 2},    # Entity 1 connectée à 0 et 2
            2: {0, 1},    # etc.
        }
    """
    
    graph = defaultdict(set)
    n = len(entities)
    
    for i in range(n):
        for j in range(i + 1, n):
            score = calculate_similarity_score(entities[i], entities[j])
            
            if score >= threshold:
                graph[i].add(j)
                graph[j].add(i)
    
    return graph


def find_connected_components(graph: Dict[int, Set[int]], n_entities: int) -> List[List[int]]:
    """
    Trouve composantes connexes (clustering via DFS)
    
    Args:
        graph: Adjacency list
        n_entities: Nombre total entities
    
    Returns: Liste de clusters [[0,1,2], [3,4], [5], ...]
    """
    
    visited = set()
    clusters = []
    
    def dfs(node: int, cluster: List[int]):
        """Depth-first search pour trouver composante"""
        visited.add(node)
        cluster.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, cluster)
    
    # Pour chaque entity
    for i in range(n_entities):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            clusters.append(cluster)
    
    return clusters


# ============================================================
# LLM ARBITRAGE CLUSTER
# ============================================================

async def arbitrate_cluster_llm(
    cluster_entities: List[Dict],
    cluster_idx: int
) -> Dict:
    """
    LLM analyse 1 cluster (max 15 entities)
    Décide: merge, keep distinct, correct types
    
    Returns:
        {
            "actions": [...],
            "reasoning": "..."
        }
    """
    
    if len(cluster_entities) == 1:
        return {"actions": [], "reasoning": "Single entity, no action needed"}
    
    # Build prompt
    entities_text = []
    for idx, e in enumerate(cluster_entities):
        # Utiliser name comme ID stable (pas numéro)
        
        entities_text.append(f"""
[{idx}]
  Name: {e['name']}
  Type: {e['type']}
  Aliases: {e.get('aliases', [])}
  Context: {e.get('context_description', 'N/A')[:120]}
""")
    
    entities_str = "\n".join(entities_text)
    
    prompt = f"""
Tu es expert en déduplication entities pour Knowledge Graph Sira.

CLUSTER #{cluster_idx} - {len(cluster_entities)} entities suspectes:

{entities_str}

TÂCHE : Décide quoi merger en utilisant uniquement les IDs numériques [0, 1, 2...].

RÈGLES CRITIQUES:
1. Si MÊME nom mais types DIFFÉRENTS:
   → MERGER + choisir type CORRECT basé sur contexte historique
   
   Exemple: "Fatima bint Muhammad" (MotherBeliever) + "Fatima bint Muhammad" (AhlBayt)
   → Fatima = fille Prophète = AhlBayt ✅ (PAS épouse = PAS MotherBeliever)
   
2. Variantes orthographiques = MERGE:
   "Aïcha bint Abi Bakr" = "Aisha bint Abu Bakr" = "A'isha"
   
3. Personnes DIFFÉRENTES malgré similarité = KEEP DISTINCT:
   "Hassan ibn Ali" ≠ "Hussain ibn Ali" (frères)
   "Aïcha" (Mère Croyants) ≠ "Fatima" (fille Prophète)

4. Si types différents (ex: Sahabi vs AhlBayt), choisis le plus précis.

5. DISTINCTION FAMILIALE : Ne confonds pas les membres d'une fratrie. 
    Deux entités avec le même père mais des prénoms différents (ex: "Asma bint Abi Bakr" et "Aïcha bint Abi Bakr") sont STRICTEMENT DISTINCTES. 
    Ne les merger JAMAIS.

PIÈGES À ÉVITER:
❌ "Aïcha bint Abi Bakr" ≠ "Fatima bint Muhammad" (personnes différentes!)
❌ Ne PAS merger basé uniquement sur context similaire

FORMAT SORTIE JSON:

{{
  "actions": [
    {{
      "type": "merge",
      "entity_ids": [2, 4],
      "canonical_name": "Nom le plus correct canoniquement",
      "canonical_type": "Type le plus précis",
      "reasoning": "Ton raisonnement..."
    }},
    {{
      "type": "keep_distinct",
      "entity_ids": [3, 9],
      "reasoning": "Deux frères différents"
    }}
  ],
  "cluster_reasoning": "Explication décision globale cluster"
}}

Si types différents dans merge, TOUJOURS spécifier canonical_type (type correct).

ANALYSE (JSON uniquement):
"""
    
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Choisir modèle selon taille cluster
    if len(cluster_entities) > 10:
        model = "gpt-4o-2024-11-20"  # Plus puissant pour gros clusters
    else:
        model = "gpt-4o-mini-2024-07-18"  # Mini suffit pour petits
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return result


def apply_cluster_decisions(entities: List[Dict], clusters: List[List[int]], llm_decisions: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    entities_to_remove = set()
    modified_entities = {}
    name_mapping = {}

    for cluster_indices, decision in zip(clusters, llm_decisions):
        for action in decision.get("actions", []):
            if action["type"] == "merge":
                ids_in_cluster = [int(i) for i in action.get("entity_ids", []) if str(i).isdigit()]                # On récupère les index réels dans la liste globale
                real_indices = [cluster_indices[i] for i in ids_in_cluster if i < len(cluster_indices)]
                
                if len(real_indices) < 2: continue
                
                to_merge = [entities[idx] for idx in real_indices]
                merged = merge_entity_group(to_merge)
                merged["name"] = action["canonical_name"]
                if "canonical_type" in action:
                    merged["type"] = action["canonical_type"]
                
                keep_idx = min(real_indices)
                modified_entities[keep_idx] = merged
                
                for idx in real_indices:
                    if idx != keep_idx:
                        entities_to_remove.add(idx)
                    # On enregistre que "Ancien Nom" -> "Nom Canonique"
                    name_mapping[entities[idx]["name"]] = action["canonical_name"]

    # Reconstruction de la liste
    new_entities = [modified_entities.get(i, e) for i, e in enumerate(entities) if i not in entities_to_remove]
    return new_entities, name_mapping


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
    graph = build_similarity_graph(entities, threshold=0.5)
    
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"     → {edge_count} edges détectées")
    
    # PHASE 2: Find clusters
    print("  🔍 Phase 2: Find connected components...")
    clusters = find_connected_components(graph, len(entities))
    
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
        
        decision = await arbitrate_cluster_llm(cluster_entities, idx)
        llm_decisions.append(decision)
        
        # Log actions
        for action in decision.get("actions", []):
            if action["type"] == "merge":
                print(f"       🔗 Merge: {len(action.get('entity_ids', []))} entities → '{action.get('canonical_name')}'")
            elif action["type"] == "keep_distinct":
                print(f"       ✅ Keep distinct: {len(action.get('entity_ids', []))} entities")
    
    # PHASE 4: Apply decisions
    print("\n  🔨 Phase 4: Apply merges...")
    entities_dedup, name_mapping = apply_cluster_decisions(entities, multi_clusters, llm_decisions)
    
    
    # PHASE 5: Update relations 
    print("  🔗 Phase 5: Update relations...")
    relations_updated = []
    for rel in relations:
        new_rel = rel.copy()
        # Si la source ou la cible a été renommée/mergée, on met à jour
        new_rel["source_entity"] = name_mapping.get(rel["source_entity"], rel["source_entity"])
        new_rel["target_entity"] = name_mapping.get(rel["target_entity"], rel["target_entity"])
        relations_updated.append(new_rel)
    
    return entities_dedup, relations_updated


def merge_entity_group(entities: List[Dict]) -> Dict:
    names = [e.get("name", "") for e in entities]
    canonical_name = max(names, key=lambda n: (len(n), "bint" in n.lower() or "ibn" in n.lower()))
    
    # Priorité du type
    current_best_type = entities[0].get("type", "Unknown")
    for e in entities:
        new_type = e.get("type", "Unknown")
        if TYPE_PRIORITY.get(new_type, 0) > TYPE_PRIORITY.get(current_best_type, 0):
            current_best_type = new_type

    # Fusion des alias et contextes
    all_aliases = []
    contexts = []
    for e in entities:
        if isinstance(e.get("aliases"), list): all_aliases.extend(e["aliases"])
        if e.get("name") != canonical_name: all_aliases.append(e["name"])
        if e.get("context_description"): contexts.append(e["context_description"])
    
    return {
        "name": canonical_name,
        "type": current_best_type,
        "aliases": list(set(all_aliases)),
        "context_description": " | ".join(set(contexts)),
        "confidence": max([e.get("confidence", 0.5) for e in entities]),
        "normalized_name": normalize_entity_name(canonical_name)
    }
#============================================================
# Clean common aliases in 3+ entities after merging
# ============================================================

def cleanup_common_aliases(entities: List[Dict]) -> List[Dict]:
    """
    Retire aliases non-distinctifs (partagés par 3+ entities)
    
    Exécuté APRÈS clustering/merges
    """
    
    # Count alias usage
    alias_counts = defaultdict(list)  # alias → [entity_names]
    
    for entity in entities:
        for alias in entity.get("aliases", []):
            normalized = normalize_entity_name(alias)
            if normalized:
                alias_counts[normalized].append(entity["name"])
    
    # Detect common aliases (3+ entities)
    common_aliases = {
        alias: entity_names 
        for alias, entity_names in alias_counts.items() 
        if len(entity_names) >= 3
    }
    
    if not common_aliases:
        return entities
    
    print(f"\n🧹 Cleanup {len(common_aliases)} aliases non-distinctifs...")
    
    for alias, entity_names in common_aliases.items():
        print(f"  ✂️  Remove '{alias}' from {len(entity_names)} entities")
    
    # Remove common aliases
    cleaned = []
    for entity in entities:
        entity_copy = entity.copy()
        
        entity_copy["aliases"] = [
            a for a in entity.get("aliases", [])
            if normalize_entity_name(a) not in common_aliases
        ]
        
        cleaned.append(entity_copy)
    
    return cleaned

# ============================================================
# VALIDATION THÉOLOGIQUE )
# ============================================================

async def validate_theology(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Valide contraintes théologiques Sira (UNCHANGED)"""
    
    print("\n🔍 Validation théologique...")
    
    errors = []
    warnings = []
    
    type_counts = defaultdict(int)
    entities_by_type = defaultdict(list)
    
    for entity in entities:
        entity_type = entity.get("type", "Unknown")
        type_counts[entity_type] += 1
        entities_by_type[entity_type].append(entity["name"])
    
    # Règle 1: 1 Prophète
    prophet_count = type_counts.get("Prophet", 0)
    if prophet_count == 0:
        errors.append("❌ Aucun Prophète détecté")
    elif prophet_count > 1:
        errors.append(f"❌ {prophet_count} Prophètes détectés (should be 1)")
        print(f"     Prophètes: {entities_by_type['Prophet']}")
    else:
        print(f"  ✅ 1 Prophète: {entities_by_type['Prophet'][0]}")
    
    # Règle 2: Max 11 Mères
    mothers_count = type_counts.get("MotherBeliever", 0)
    if mothers_count > 11:
        errors.append(f"❌ {mothers_count} Mères détectées (max 11)")
        print(f"     Mères: {entities_by_type['MotherBeliever']}")
    elif mothers_count > 0:
        print(f"  ✅ {mothers_count} Mères des Croyants")
    
    # Règle 3: AhlBayt (avec fuzzy match)
    VALID_AHL_BAYT = [
        "fatima bint muhammad",
        "ali ibn abi talib",
        "hassan ibn ali",
        "hussain ibn ali",
        "hamza ibn abdul muttalib",
        "abbas ibn abdul muttalib",
        "abu talib",
    ]
    
    ahl_bayt = entities_by_type.get("AhlBayt", [])
    for member in ahl_bayt:
        normalized = normalize_entity_name(member)
        
        is_valid = False
        for valid_member in VALID_AHL_BAYT:
            if similarity(normalized, valid_member) > 0.85:
                is_valid = True
                break
        
        if not is_valid:
            warnings.append(f"⚠️  AhlBayt suspect: {member}")
    
    if ahl_bayt:
        print(f"  ℹ️  {len(ahl_bayt)} AhlBayt members")
    
    stats = {
        "total_entities": len(entities),
        "prophets": prophet_count,
        "mothers": mothers_count,
        "sahaba": type_counts.get("Sahabi", 0),
        "ahl_bayt": len(ahl_bayt),
        "places": type_counts.get("Place", 0),
        "battles": type_counts.get("Battle", 0),
    }
    
    print(f"\n📊 Stats extraction:")
    for key, value in stats.items():
        print(f"  {key:20s}: {value}")
    
    if errors:
        print(f"\n❌ ERREURS ({len(errors)}):")
        for err in errors:
            print(f"  {err}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"  {warn}")
    
    valid = len(errors) == 0
    
    if valid:
        print("\n✅ Validation théologique PASSED\n")
    else:
        print("\n❌ Validation théologique FAILED\n")
    
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


# ============================================================
# PIPELINE COMPLÈTE
# ============================================================

async def post_process_graph_extraction(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Pipeline post-processing complète"""
    
    print(f"\n{'='*70}")
    print("🔧 POST-PROCESSING GRAPH")
    print(f"{'='*70}")
    
    # 1. Deduplication CLUSTERING
    entities_dedup, relations_updated = await deduplicate_entities_clustered(
        entities,
        relations
    )
    # 2. Alias cleanup (simple, déterministe)
    entities_clean = cleanup_common_aliases(entities_dedup)
    
    # 3. Validation théologique
    validation = await validate_theology(entities_clean)
    
    # 4. Clean orphaned relations
    entity_names = {e["name"] for e in entities_clean}
    
    relations_clean = [
        rel for rel in relations_updated
        if (rel["source_entity"] in entity_names and
            rel["target_entity"] in entity_names)
    ]
    
    orphan_count = len(relations_updated) - len(relations_clean)
    if orphan_count > 0:
        print(f"\n⚠️  {orphan_count} relations orphelines supprimées")
    
    print(f"✅ Relations nettoyées: {len(relations)} → {len(relations_clean)}")
    
    print(f"\n{'='*70}")
    print("✅ POST-PROCESSING TERMINÉ")
    print(f"{'='*70}\n")
    
    return entities_clean, relations_clean, validation