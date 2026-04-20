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
    ANCHORING_RESOLUTION_USER_PROMPT
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

    def __init__(self, llm_service: LLMService):
        """Initializes the resolver with a LLM service for semantic decision-making."""
        self.llm_service = llm_service


    async def llm_resolve(self, entities: List[EntityModel]) -> Tuple[List[EntityModel], Dict[str, str]]:
        """
        Coordinates the high-level workflow for semantic resolution.
        
        This method first settles ambiguities with the reference Encyclopedia (Anchoring) 
        and then attempts to merge remaining unknown entities through hybrid clustering.
        
        Returns:
            - resolved_entities: List of EntityModel with updated canonical_ids.
            - all_llm_mappings: Dict mapping local titles to their resolved target names/IDs.
        """
        all_llm_mappings = {}
        
        # 1. ENCYCLOPEDIA ANCHORING
        # Handles cases where the CoreResolver flagged multiple potential matches.
        ambiguous_entities = [e for e in entities if e.attributes.get("anchoring_candidates")]
        
        if ambiguous_entities:
            logger.info(f"⚓ Resolving anchoring for {len(ambiguous_entities)} ambiguous entities...")
            tasks = [self._resolve_anchoring(e) for e in ambiguous_entities]
            results = await asyncio.gather(*tasks)
            
            for entity, result in zip(ambiguous_entities, results):
                choice = result.get("choice") #it should be an uuid

                if choice and choice != "NEW_ENTITY":
                    all_llm_mappings[entity.id] = choice
                    entity.canonical_id = choice
                    entity.review_status = "LLM_VALIDATED"

        # 2. SEMANTIC CLUSTERING (Orphan Entities)
        # Groups entities that didn't match the Encyclopedia but might be duplicates of each other.
        orphans = [e for e in entities if not e.canonical_id]
        
        if orphans:
            logger.info(f"🧩 Clustering {len(orphans)} orphan entities for semantic resolution...")

            # We group by category using a simple dictionary approach
            orphans_by_cat = {}
            for e in orphans:
                orphans_by_cat.setdefault(e.category, []).append(e)

            for category, group in orphans_by_cat.items():
                clusters = self._create_hybrid_clusters(group) 
                for cluster in clusters:
                    # Pyramidal resolution handles merging if cluster >= 2, 
                    # otherwise no action is needed
                    if len(cluster) >= 2:
                        mapping = await self._pyramidal_resolve(cluster, str(category))
                        all_llm_mappings.update(mapping)
                    else:
                        # Single entity in a cluster requires no LLM action
                        continue

        return entities, all_llm_mappings

    async def _resolve_cluster(self, cluster: List[EntityModel], entity_category: str) -> Dict[str, str]:
        """
        Analyzes a semantic cluster via LLM to identify internal duplicates.
        
        Uses a 'tuple-based' prompt where the LLM returns MERGE instructions.
        The method leverages EntityModel objects, ensuring consistent access to 
        titles and descriptions for the resolution process.
        """
        
        if len(cluster) < 2:
            return {}

        # Context optimization: Prepare snippets from EntityModel descriptions.
        # We ensure description is clean and truncated to keep focus on identifiers.
        candidates_list = []
        for entity in cluster:
            # Using clean string formatting from EntityModel
            clean_context = entity.description.replace("\n", " ").strip()
            snippet = (clean_context[:250] + "...") if len(clean_context) > 300 else clean_context
            candidates_list.append(f"Title: {entity.title} (Type: {entity.type}): {snippet}")

        candidates_text = "\n".join(candidates_list)

        try:
            # Expects tuples like ["MERGE", "Source_title", "Target_title"]
            title_to_id = {entity.title: entity.id for entity in cluster}

            tuples = await self.llm_service.ask_tuples(
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
                    source_title, target_title = t[1].strip(), t[2].strip()
                    
                    # Traduction Title -> ID
                    if source_title in title_to_id and target_title in title_to_id:
                        mapping[title_to_id[source_title]] = title_to_id[target_title]
                    else:
                        logger.warning(f"⚠️ LLM a retourné un titre inconnu: {source_title} -> {target_title}")
            
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
            slug_to_id = {c.slug: c.id for c in candidates}

            result = await self.llm_service.ask_json(
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
                    entity.canonical_id = slug_to_id[choice] # Injection of the UUID
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
            
            # Emergency break to prevent infinite loops
            if len(merged_away) == 0:
                break

        return global_mappings

    def _update_transitive_mappings(self, master: Dict[str, str], new: Dict[str, str]):
        """
        Updates the master mapping dictionary ensuring transitive integrity.
        Logic: If A -> B already exists and we find B -> C, update A to point to C.
        """
        # 1. Integrate new findings into the master dictionary
        for source, target in new.items():
            master[source] = target
        
        # 2. Resolve multi-hop mappings (Transitivity)
        # We iterate to ensure all paths lead to the final canonical representative
        for source in list(master.keys()):
            path = [] # Track path to detect circular references
            target = master[source]
            
            while target in master and target not in path:
                path.append(target)
                target = master[target]
            
            master[source] = target


    def _create_hybrid_clusters(self, entities: List[EntityModel]) -> List[List[EntityModel]]:
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
        # We temporarily use a DataFrame here because TF-IDF/Cosine operations are highly optimized for this format.

        df = pd.DataFrame([e.model_dump() for e in entities])
        texts = df.apply(lambda x: f"{x['title']} {str(x['description']).replace('|', ' ')}", axis=1).tolist()
        
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
        
        # Return list of lists of EntityModels
        return [[entities[idx] for idx in component] for component in clusters]

    
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