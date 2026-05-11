import pandas as pd
import asyncio
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List

from app.core.data_model.entity import EntityModel
from app.core.data_model.encyclopedia import EncyclopediaEntry

from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT, 
    ENTITY_RESOLUTION_USER_PROMPT,
    ANCHORING_RESOLUTION_SYSTEM_PROMPT,
    ANCHORING_RESOLUTION_USER_PROMPT,
    CONSULTANT_RESOLUTION_SYSTEM_PROMPT,
    CONSULTANT_RESOLUTION_USER_PROMPT
)
from app.core.config.graph_config import MAX_CLUSTER_BATCH
from app.indexing.operations.text.text_utils import similarity

import logging

logger = logging.getLogger(__name__)

class LLMResolver:
    """
    Advanced entity resolution engine using Large Language Models to handle complex ambiguities.
    
    This resolver operates in two main modes:
    1. Anchoring Resolution: Selecting the correct record when multiple Encyclopedia 
       matches are found for a single entity.
    2. Pyramidal Clustering: Identifying semantic duplicates among 'orphan' entities 
       (those not found in the Encyclopedia) by analyzing names and context snippets.
    """

    def __init__(self, light_service: LLMService, heavy_service: LLMService):
        """Initializes the resolver with both LLM services for semantic decision-making."""
        self.light_service = light_service
        self.heavy_service = heavy_service


    async def llm_resolve(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Coordinates the high-level workflow using a cascade approach:
        1. Encyclopedia Anchoring (Reference matching).
        2. Algorithmic Clustering (Structural matching + Singletons).
        3. LLM Consultant (Semantic bridges between clusters).
        4. Pyramidal Resolution (Final verdict on merged super-clusters).

        Args:
            - entities: The Dataframe of entities to resolve.
        
        Returns:
            - resolved_entities : List of EntityModel with updated canonical_ids.
            - all_llm_mappings : Dict mapping local titles to their resolved target names/IDs.
        """
        all_llm_mappings = {}
        entity_models = [EntityModel(**r) for r in entities.to_dict('records')]
        
        # 1. ENCYCLOPEDIA ANCHORING
        ambiguous_entities = [e for e in entity_models if e.attributes.get("anchoring_candidates")]
        if ambiguous_entities:
            tasks = [self._resolve_anchoring(e) for e in ambiguous_entities]
            await asyncio.gather(*tasks)
            for entity in ambiguous_entities:
                if entity.canonical_id:
                    all_llm_mappings[entity.id] = entity.canonical_id

        # 2. SEMANTIC CLUSTERING (Orphan Entities)
        orphans = [e for e in entity_models if not e.canonical_id]
        
        if orphans:
            orphans_by_cat = {}
            for e in orphans:
                orphans_by_cat.setdefault(e.category, []).append(e)

            for category, group in orphans_by_cat.items():
                logger.info(f"🔍 Analyzing category '{category}' with {len(group)} entities...")

                # A. STEP 1: Algorithmic Clustering (Blocking)
                # Creates clusters based on typos/fuzzy match, including singletons.
                algo_clusters = self._create_algo_clusters(group)
                
                # B. STEP 2: LLM Consultant (Semantic Bridge)
                # We pick the first entity of each cluster as a 'Representative'.
                representatives = [cluster[0] for cluster in algo_clusters]
                
                # The consultant returns indices of 'representatives' that should merge.
                # Example: [[0, 2]] means algo_clusters[0] and algo_clusters[2] are suspected duplicates.
                bridge_indices_list = await self._get_semantic_bridges(representatives, str(category))
                
                # C. STEP 3: Super-Clustering
                # We merge algo_clusters together based on LLM Consultant's advice.
                final_clusters_to_resolve = self._merge_clusters_by_indices(algo_clusters, bridge_indices_list)
                
                logger.info(f"🚀 Processing {len(final_clusters_to_resolve)} refined clusters for {category}...")

                # D. STEP 4: Pyramidal Resolver (Final Verdict)
                for cluster in final_clusters_to_resolve:
                    if len(cluster) >= 2:
                        # The resolver now sees the full context of the merged groups.
                        mapping = await self._pyramidal_resolve(cluster, str(category))
                        all_llm_mappings.update(mapping)

        resolved_entities = pd.DataFrame([e.model_dump() for e in entity_models])
        return resolved_entities, all_llm_mappings
    
    async def _resolve_cluster(self, cluster: List[EntityModel], entity_category: str) -> Dict[str, str]:
        """
        Analyzes a semantic cluster via LLM to identify internal duplicates.
        
        Uses a 'tuple-based' prompt where the LLM returns MERGE instructions.
        The method leverages EntityModel objects, ensuring consistent access to 
        titles and descriptions for the resolution process.
        """
        
        if len(cluster) < 2:
            return {}

        # Use index-based mapping to prevent LLM hallucination of titles
        index_to_id = {str(i): entity.id for i, entity in enumerate(cluster)}

        # Context optimization: Prepare snippets from EntityModel descriptions.
        # We ensure description is clean and truncated to keep focus on identifiers.
        candidates_list = []
        for i, entity in enumerate(cluster):
            # Using clean string formatting from EntityModel
            clean_context = entity.description.replace("\n", " ").strip()
            snippet = (clean_context[:250] + "...") if len(clean_context) > 300 else clean_context
            # Injecting the index [#i] for the LLM to reference
            candidates_list.append(f"[#{i}] Title: {entity.title} (Type: {entity.type}): {snippet}")

        candidates_text = "\n".join(candidates_list)

        try:
            # Expects tuples like ["MERGE", "0", "1"] where numbers are the indices
            tuples = await self.heavy_service.ask_tuples(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ENTITY_RESOLUTION_USER_PROMPT.format(
                    entity_type=entity_category,
                    candidates=candidates_text
                )
            )

            # Map the merges based on LLM output 
            mapping = {}
            for t in tuples:
                if len(t) >= 3 and t[0].upper() == "MERGE":
                    # Clean potential brackets or hash symbols the LLM might add
                    src_idx = t[1].replace("[", "").replace("]", "").replace("#", "").strip()
                    tgt_idx = t[2].replace("[", "").replace("]", "").replace("#", "").strip()
                    
                    # Translation Index -> ID
                    if src_idx in index_to_id and tgt_idx in index_to_id:
                        # Prevent self-merging just in case
                        if src_idx != tgt_idx:
                            mapping[index_to_id[src_idx]] = index_to_id[tgt_idx]
                    else:
                        logger.warning(f"⚠️ LLM returned an unknown index: {src_idx} -> {tgt_idx}")
            
            return mapping

        except Exception as e:
            # We log the error but return an empty dict to allow the pipeline to continue
            logger.error(f"❌ LLMResolver cluster error ({entity_category}): {e}")
            return {}

    async def _resolve_anchoring(self, entity: EntityModel) -> Dict[str, Any]:
        """
        Resolves ambiguity when an entity matches multiple Encyclopedia entries.
        
        The LLM acts as a discriminator, comparing the entity's current context 
        against the summaries of reference candidates to pick the correct ID 
        or declare it a 'NEW_ENTITY'.
        """
        # Retrieve candidates stored in the entity's attributes during the Core resolution phase
        candidates = entity.attributes.get("anchoring_candidates", [])
        if not candidates:
            return {"choice": "NEW_ENTITY"}

        # Format candidates for the LLM prompt using our helper
        candidates_text = self._format_anchoring_candidates(candidates)
        
        # Prepare entity context: truncate description to fit LLM window effectively
        entity_context = entity.description.replace("\n", " ").strip()[:250]

        try:
            # Standardized JSON response for direct programmatic integration
            # The LLM evaluates the context vs candidates to determine the best match
            slug_to_id = {c.get("slug"): c.get("id") for c in candidates}

            result = await self.light_service.ask_json(
                system_prompt=ANCHORING_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ANCHORING_RESOLUTION_USER_PROMPT.format(
                    entity_title=entity.title,
                    entity_type=entity.type,
                    entity_context=entity_context,
                    candidates_text=candidates_text
                ))
            
            choice = result.get("choice")
            
            if choice and choice != "NEW_ENTITY":
                if choice in slug_to_id:
                    entity.canonical_id = slug_to_id[choice] 
                    entity.review_status = "LLM_VALIDATED"
                    # TODO : entity.attributes["llm_confidence"] = result.get("confidence") ,in the future, we want to have a condiance score of the LLM decisions

                else:
                    logger.warning(f"⚠️ Le LLM a renvoyé un slug inconnu : {choice}")
                    entity.review_status = "NOT_KNOWN"
            else:
                entity.review_status = "NOT_KNOWN"

            return result
            
        except Exception as e:
            # Fallback to NEW_ENTITY in case of LLM communication failure
            logger.error(f"❌ LLMResolver anchoring error ({entity.title}): {e}")
            return {"choice": "NEW_ENTITY"}

    async def _pyramidal_resolve(self, cluster: List[EntityModel], entity_category: str) -> Dict[str, str]:
        """
        Processes large clusters by batches and recursively re-evaluates 
        survivors until a stable set is reached.
        """
        # We work with a list of EntityModels instead of a DataFrame
        current_pool = list(cluster)
        global_mappings = {}

        while len(current_pool) > 1:
            # Case 1: Cluster fits in a single batch, process and exit
            if len(current_pool) <= MAX_CLUSTER_BATCH:
                final_mapping = await self._resolve_cluster(current_pool, entity_category)
                self._update_transitive_mappings(global_mappings, final_mapping)
                break

            # Case 2: Cluster is too large, split into batches and process in parallel
            # We use standard list slicing
            batches = [
                current_pool[i : i + MAX_CLUSTER_BATCH] 
                for i in range(0, len(current_pool), MAX_CLUSTER_BATCH)
            ]
            
            logger.info(f"🚀 Pyramidal Round: Processing {len(batches)} batches for {len(current_pool)} entities...")
            
            # Execute all batches concurrently
            results = await asyncio.gather(*[self._resolve_cluster(b, entity_category) for b in batches])
            
            # Merge results from the current round
            round_mappings = {}
            for m in results:
                round_mappings.update(m)

            # Exit if no further merges are identified by the LLM
            if not round_mappings:
                break 

            # Update global registry with new findings (maintaining transitivity)
            self._update_transitive_mappings(global_mappings, round_mappings)

            # Prepare for next round: keep only 'surviving' entities
            # (Entities that were NOT the source of a merge, i.e., keys in round_mappings)
            merged_away = set(round_mappings.keys())
            current_pool = [e for e in current_pool if e.id not in merged_away]

        return global_mappings

    def _update_transitive_mappings(self, master: Dict[str, str], new: Dict[str, str]):
        """
        Updates the master mapping dictionary ensuring transitive integrity.
        Logic: If A -> B already exists and we find B -> C, update A to point to C.
        """

        # 1. Merge newly discovered mappings
        master.update(new)

        # 2. Resolve transitive chains with path compression
        for source in master:
            target = master[source]

            # If the target also redirects somewhere else, follow the chain until the final canonical node
            if target in master:
                visited = {source}  # Prevent circular references

                while target in master and target not in visited:
                    visited.add(target)
                    target = master[target]

                # Compress the path: directly point source -> final canonical target
                master[source] = target
        
    def _create_algo_clusters(self, entities: List[EntityModel]) -> List[List[EntityModel]]:
        """
        Groups entities into 'potential duplicate clusters' using a graph-based approach.
        
        Algorithm:
        1. Nodes = Entities.
        2. Edges = Created if two entities share semantic similarity (TF-IDF),
           structural similarity (Levenshtein), or name containment.
        3. Clusters = Connected components of the resulting graph.
        """
        if len(entities) <= 1:
            return [entities]

        # 1. Prepare data for vectorized comparison

        texts = [f"{e.title} {str(e.description).replace('|', ' ')}" for e in entities]

        vectorizer = TfidfVectorizer(stop_words='english') 
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)

        # 2. Build the graph based on semantic and structural criteria
        G = nx.Graph()
        G.add_nodes_from(range(len(entities)))

        for i in range(len(entities)):
            slug_i = entities[i].slug
            for j in range(i + 1, len(entities)):
                slug_j = entities[j].slug

                # Similarity criterias

                is_semantic = cosine_sim[i, j] >= 0.3         
                is_contained = (slug_i in slug_j or slug_j in slug_i) if (len(slug_i) > 4 and len(slug_j) > 4) else False     
                is_fuzzy = similarity(slug_i, slug_j) >= 0.75

                if is_semantic or is_contained or is_fuzzy:
                    G.add_edge(i, j)

        # 3. Extract connected components
        clusters = list(nx.connected_components(G))
        
        logger.debug(f"📊 Hybrid clustering created {len(clusters)} groups from {len(entities)} entities.")
        
        # Return list of lists of EntityModels, sorted to make the first elem the semantically richest entity
        return [
    sorted(
        [entities[idx] for idx in component], 
        key=lambda e: (len(e.title), len(e.description)), 
        reverse=True
    ) 
    for component in clusters
]

    async def _get_semantic_bridges(self, representatives: List[EntityModel], category: str) -> List[List[int]]:
        """
        Consults the LLM to find semantic overlaps between cluster representatives.
        Returns a list of index groups, e.g., [[0, 2], [1, 3, 4]].
        """
        if len(representatives) <= 1:
            return []

        titles_text = "\n".join([f"[#{i}] {e.title}" for i, e in enumerate(representatives)])

        try:
            # Using the standardized CONSULTANT prompts
            bridges = await self.light_service.ask_json(
                system_prompt=CONSULTANT_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=CONSULTANT_RESOLUTION_USER_PROMPT.format(
                    category=category,
                    titles_text=titles_text
                )
            )
            return bridges if isinstance(bridges, list) else []
        except Exception as e:
            logger.error(f"❌ LLMConsultant bridge error: {e}")
            return []
        
    def _merge_clusters_by_indices(self, clusters: List[List[EntityModel]], bridge_indices: List[List[int]]) -> List[List[EntityModel]]:
        """
        Merges existing clusters into larger ones based on bridge suggestions.
        Uses a graph-based approach to handle transitive merges (A=B, B=C -> [A,B,C]).
        """
        if not bridge_indices:
            return clusters

        # Build a meta-graph of clusters
        meta_G = nx.Graph()
        meta_G.add_nodes_from(range(len(clusters)))

        for bridge in bridge_indices:
            for i in range(len(bridge)):
                for j in range(i + 1, len(bridge)):
                    meta_G.add_edge(bridge[i], bridge[j])

        # Extract connected components of clusters
        meta_components = list(nx.connected_components(meta_G))
        
        new_clusters = []
        for component in meta_components:
            # Flatten all entities from the merged clusters into one
            merged_cluster = []
            for cluster_idx in component:
                merged_cluster.extend(clusters[cluster_idx])
            new_clusters.append(merged_cluster)

        return new_clusters

    def _format_anchoring_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Formats the list of EncyclopediaEntry candidates into a structured string for the LLM prompt.
        
        Uses the model's dictionary representation to ensure all relevant fields (id, title, summary) 
        are correctly extracted for semantic comparison.
        """
        formatted_list = []
        for c in candidates:
            # We use dict access as candidates are usually passed as model_dump() dicts
            slug = c.get("slug", "UNKNOWN")
            title = c.get("title", "UNKNOWN")
            summary = c.get("core_summary", "No summary available.")

            props = c.get("properties", {})
            props_str = " | ".join([f"{k}: {v}" for k, v in props.items()])
            
            formatted_list.append(
                f"- SLUG: {slug} | Name: {title} | Summary: {summary} | Properties: {props_str}"
            )
            
        return "\n".join(formatted_list)