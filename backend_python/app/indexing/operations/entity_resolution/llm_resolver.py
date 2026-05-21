import pandas as pd
import asyncio
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List

from app.core.data_model.entity import EntityModel
from app.core.data_model.encyclopedia import EncyclopediaEntry

from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts.entity_resolution_prompts import (
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
    Advanced entity resolution engine using specialized, task-specific Large Language Models 
    to handle complex ambiguities and multi-stage deduplication.
    
    This resolver operates in three main semantic layers:
    1. Anchoring Resolution: Selecting the correct record when multiple Encyclopedia 
       matches are found for a single entity context.
    2. Algorithmic Blocking & Consultant Bridging: Grouping orphan entities and using a specialized
       prosopographical analyzer to spot semantic aliases.
    3. Pyramidal Final Resolution: Executing deep context batch analysis to output structural merges.
    """

    def __init__(
        self, 
        entity_resolution_service: LLMService, 
        anchoring_resolution_service: LLMService, 
        consultant_resolution_service: LLMService
    ):
        """Initializes the resolver with decoupled fine-grained task LLM services."""
        self.entity_resolution_service = entity_resolution_service
        self.anchoring_resolution_service = anchoring_resolution_service
        self.consultant_resolution_service = consultant_resolution_service

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
                algo_clusters = self._create_algo_clusters(group)
                
                # B. STEP 2: LLM Consultant (Semantic Bridge)
                representatives = [cluster[0] for cluster in algo_clusters]
                bridge_indices_list = await self._get_semantic_bridges(representatives, str(category))
                
                # C. STEP 3: Super-Clustering
                final_clusters_to_resolve = self._merge_clusters_by_indices(algo_clusters, bridge_indices_list)
                
                logger.info(f"🚀 Processing {len(final_clusters_to_resolve)} refined clusters for {category}...")

                # D. STEP 4: Pyramidal Resolver (Final Verdict)
                for cluster in final_clusters_to_resolve:
                    if len(cluster) >= 2:
                        mapping = await self._pyramidal_resolve(cluster, str(category))
                        all_llm_mappings.update(mapping)

        resolved_entities = pd.DataFrame([e.model_dump() for e in entity_models])
        return resolved_entities, all_llm_mappings
    
    async def _resolve_cluster(self, cluster: List[EntityModel], entity_category: str) -> Dict[str, str]:
        """
        Analyzes a semantic cluster via LLM to identify internal duplicates.
        Utilizes the specialized entity resolution service (optimized for strict merge formatting).
        """
        if len(cluster) < 2:
            return {}

        index_to_id = {str(i): entity.id for i, entity in enumerate(cluster)}

        candidates_list = []
        for i, entity in enumerate(cluster):
            clean_context = entity.description.replace("\n", " ").strip()
            snippet = (clean_context[:250] + "...") if len(clean_context) > 300 else clean_context
            candidates_list.append(f"[#{i}] Title: {entity.title} (Type: {entity.type}): {snippet}")

        candidates_text = "\n".join(candidates_list)

        try:
            tuples = await self.entity_resolution_service.ask_tuples(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ENTITY_RESOLUTION_USER_PROMPT.format(
                    entity_type=entity_category,
                    candidates=candidates_text
                )
            )
            
            mapping = {}
            for t in tuples:
                if len(t) >= 3 and t[0].upper() == "MERGE":
                    src_idx = t[1].replace("[", "").replace("]", "").replace("#", "").strip()
                    tgt_idx = t[2].replace("[", "").replace("]", "").replace("#", "").strip()
                    
                    if src_idx in index_to_id and tgt_idx in index_to_id:
                        if src_idx != tgt_idx:
                            mapping[index_to_id[src_idx]] = index_to_id[tgt_idx]
                    else:
                        logger.warning(f"⚠️ LLM returned an unknown index: {src_idx} -> {tgt_idx}")
            
            return mapping

        except Exception as e:
            logger.error(f"❌ LLMResolver cluster error ({entity_category}): {e}")
            return {}

    async def _resolve_anchoring(self, entity: EntityModel) -> Dict[str, Any]:
        """
        Resolves ambiguity when an entity matches multiple Encyclopedia entries.
        Utilizes the anchoring resolution service.
        """
        candidates = entity.attributes.get("anchoring_candidates", [])
        if not candidates:
            return {"choice": "NEW_ENTITY"}

        candidates_text = self._format_anchoring_candidates(candidates)
        entity_context = entity.description.replace("\n", " ").strip()[:250]

        try:
            slug_to_id = {c.get("slug"): c.get("id") for c in candidates}

            result = await self.anchoring_resolution_service.ask_json(
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
                else:
                    logger.warning(f"⚠️ Le LLM a renvoyé un slug inconnu : {choice}")
                    entity.review_status = "NOT_KNOWN"
            else:
                entity.review_status = "NOT_KNOWN"

            return result
            
        except Exception as e:
            logger.error(f"❌ LLMResolver anchoring error ({entity.title}): {e}")
            return {"choice": "NEW_ENTITY"}

    async def _get_semantic_bridges(self, representatives: List[EntityModel], category: str) -> List[List[int]]:
        """
        Consults the LLM to find semantic historical overlaps between cluster representatives.
        Utilizes the specialized consultant service for prosopographical clustering analysis.
        """
        if len(representatives) <= 1:
            return []

        titles_text = "\n".join([f"[#{i}] {e.title}" for i, e in enumerate(representatives)])

        try:
            bridges = await self.consultant_resolution_service.ask_json(
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

    # ==============================================================================
    # ⚙️ GRAPH UTILS & ALGORITHMIC BLOCKING (No LLM structural changes required)
    # ==============================================================================

    async def _pyramidal_resolve(self, cluster: List[EntityModel], entity_category: str) -> Dict[str, str]:
        """Processes large clusters by batches and recursively re-evaluates survivors."""
        current_pool = list(cluster)
        global_mappings = {}

        while len(current_pool) > 1:
            if len(current_pool) <= MAX_CLUSTER_BATCH:
                final_mapping = await self._resolve_cluster(current_pool, entity_category)
                self._update_transitive_mappings(global_mappings, final_mapping)
                break

            batches = [
                current_pool[i : i + MAX_CLUSTER_BATCH] 
                for i in range(0, len(current_pool), MAX_CLUSTER_BATCH)
            ]
            
            logger.info(f"🚀 Pyramidal Round: Processing {len(batches)} batches for {len(current_pool)} entities...")
            results = await asyncio.gather(*[self._resolve_cluster(b, entity_category) for b in batches])
            
            round_mappings = {}
            for m in results:
                round_mappings.update(m)

            if not round_mappings:
                break 

            self._update_transitive_mappings(global_mappings, round_mappings)
            merged_away = set(round_mappings.keys())
            current_pool = [e for e in current_pool if e.id not in merged_away]

        return global_mappings

    def _update_transitive_mappings(self, master: Dict[str, str], new: Dict[str, str]):
        """Updates the master mapping dictionary ensuring transitive integrity and path compression."""
        master.update(new)
        for source in master:
            target = master[source]
            if target in master:
                visited = {source}
                while target in master and target not in visited:
                    visited.add(target)
                    target = master[target]
                master[source] = target
        
    def _create_algo_clusters(self, entities: List[EntityModel]) -> List[List[EntityModel]]:
        """Groups entities into potential duplicate clusters using a fuzzy-graph approach."""
        if len(entities) <= 1:
            return [entities]

        texts = [f"{e.title} {str(e.description).replace('|', ' ')}" for e in entities]
        vectorizer = TfidfVectorizer(stop_words='english') 
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)

        G = nx.Graph()
        G.add_nodes_from(range(len(entities)))

        for i in range(len(entities)):
            slug_i = entities[i].slug
            for j in range(i + 1, len(entities)):
                slug_j = entities[j].slug

                is_semantic = cosine_sim[i, j] >= 0.3         
                is_contained = (slug_i in slug_j or slug_j in slug_i) if (len(slug_i) > 4 and len(slug_j) > 4) else False     
                is_fuzzy = similarity(slug_i, slug_j) >= 0.75

                if is_semantic or is_contained or is_fuzzy:
                    G.add_edge(i, j)

        clusters = list(nx.connected_components(G))
        return [
            sorted([entities[idx] for idx in component], key=lambda e: (len(e.title), len(e.description)), reverse=True) 
            for component in clusters
        ]

    def _merge_clusters_by_indices(self, clusters: List[List[EntityModel]], bridge_indices: List[List[int]]) -> List[List[EntityModel]]:
        """Merges existing clusters into larger ones based on bridge suggestions via a meta-graph."""
        if not bridge_indices:
            return clusters

        meta_G = nx.Graph()
        meta_G.add_nodes_from(range(len(clusters)))

        for bridge in bridge_indices:
            for i in range(len(bridge)):
                for j in range(i + 1, len(bridge)):
                    meta_G.add_edge(bridge[i], bridge[j])

        meta_components = list(nx.connected_components(meta_G))
        new_clusters = []
        for component in meta_components:
            merged_cluster = []
            for cluster_idx in component:
                merged_cluster.extend(clusters[cluster_idx])
            new_clusters.append(merged_cluster)

        return new_clusters

    def _format_anchoring_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Formats the list of EncyclopediaEntry candidates into a structured string for the prompt."""
        formatted_list = []
        for c in candidates:
            slug = c.get("slug", "UNKNOWN")
            title = c.get("title", "UNKNOWN")
            summary = c.get("core_summary", "No summary available.")
            props = c.get("properties", {})
            props_str = " | ".join([f"{k}: {v}" for k, v in props.items()])
            formatted_list.append(f"- SLUG: {slug} | Name: {title} | Summary: {summary} | Properties: {props_str}")
        return "\n".join(formatted_list)
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