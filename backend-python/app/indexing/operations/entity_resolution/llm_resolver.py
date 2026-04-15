import pandas as pd
import asyncio
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple
from app.services.llm.service import LLMService
from app.core.prompts.graph_prompts import (
    ENTITY_RESOLUTION_SYSTEM_PROMPT, 
    ENTITY_RESOLUTION_USER_PROMPT,
    ANCHORING_RESOLUTION_SYSTEM_PROMPT,
    ANCHORING_RESOLUTION_USER_PROMPT
)
from app.core.config.graph_config import MAX_CLUSTER_BATCH
from app.indexing.operations.text.text_utils import similarity, normalize_entity_name

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


    async def resolve_complex_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Coordinates the high-level workflow for semantic resolution.
        
        This method first settles ambiguities with the reference Encyclopedia (Anchoring) 
        and then attempts to merge remaining unknown entities through hybrid clustering.
        
        Returns:
            - df: Updated DataFrame with canonical_id assignments.
            - all_llm_mappings: Dict mapping local names to their resolved target names/IDs.
        """
        all_llm_mappings = {}
        
        # 1. ENCYCLOPEDIA ANCHORING
        # Handles cases where the CoreResolver flagged multiple potential matches.
        if "anchoring_candidates" in df.columns:
            ambiguous_df = df[df["anchoring_candidates"].notna()]
            if not ambiguous_df.empty:
                logger.info(f"⚓ Resolving anchoring for {len(ambiguous_df)} ambiguous entities...")
                tasks = [self.resolve_anchoring(row) for _, row in ambiguous_df.iterrows()]
                results = await asyncio.gather(*tasks)
                
                for idx, result in zip(ambiguous_df.index, results):
                    choice = result.get("choice")
                    if choice and choice != "NEW_ENTITY":
                        old_name = df.at[idx, "title"]
                        all_llm_mappings[old_name] = choice
                        df.at[idx, "canonical_id"] = choice

        # 2. SEMANTIC CLUSTERING (Orphan Entities)
        # Groups entities that didn't match the Encyclopedia but might be duplicates of each other.
        orphans = df[df["canonical_id"].isna()].copy()
        if not orphans.empty:
            logger.info(f"🧩 Clustering {len(orphans)} orphan entities for semantic resolution...")
            for entity_type, type_group in orphans.groupby("type"):
                # Use Graph theory and NLP to find potential duplicate groups
                clusters = self._create_hybrid_clusters(type_group) 
                
                for cluster in clusters:
                    # Batch processing to respect LLM context window limits (and avoid hallucinations)
                    if len(cluster) > MAX_CLUSTER_BATCH: #TODO add a recursive methode to handle big clusters like this : 100 --> 4 x 25 to process, then we have 4 x 12, we aggregate --> 2 x 24 , we process ...
                        for i in range(0, len(cluster), MAX_CLUSTER_BATCH):
                            batch = cluster.iloc[i:i + MAX_CLUSTER_BATCH]
                            mapping = await self.resolve_cluster(batch, str(entity_type))
                            all_llm_mappings.update(mapping)
                    else:
                        mapping = await self.resolve_cluster(cluster, str(entity_type))
                        all_llm_mappings.update(mapping)

        return df, all_llm_mappings

    async def resolve_cluster(self, cluster_df: pd.DataFrame, entity_type: str) -> Dict[str, str]:
        """
        Analyzes a semantic cluster via LLM to identify internal duplicates.
        
        Uses a 'tuple-based' prompt where the LLM returns MERGE instructions.
        Descriptions are truncated to ensure maximum information density 
        within the prompt's context window.
        """
        
        if len(cluster_df) < 2:
            return {}

        # Context optimization: merge list of descriptions and truncate to keep focus on identifiers
        candidates_list = []
        for row in cluster_df.itertuples():
            full_context = " ".join(set(row.description)) if isinstance(row.description, list) else str(row.description)
            clean_context = full_context.replace("\n", " ").strip()
            snippet = clean_context[:250] + "..." if len(clean_context) > 300 else clean_context
            candidates_list.append(f"- {row.title} (Type: {entity_type}): {snippet}")

        candidates_text = "\n".join(candidates_list)

        try:
            # Expects tuples like ["MERGE", "Source_Name", "Target_Name"]
            tuples = await self.llm_service.ask_tuples(
                system_prompt=ENTITY_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ENTITY_RESOLUTION_USER_PROMPT.format(
                    entity_type=entity_type,
                    candidates=candidates_text
                )
            )

            mapping = {t[1].strip(): t[2].strip() for t in tuples if len(t) >= 3 and t[0].upper() == "MERGE"}
            return mapping

        except Exception as e:
            logger.error(f"❌ LLMResolver cluster error ({entity_type}): {e}")
            return {}
        

    async def resolve_anchoring(self, row: pd.Series) -> Dict[str, Any]:
        """
        Resolves ambiguity when an entity matches multiple Encyclopedia entries.
        
        The LLM acts as a discriminator, comparing the entity's current context 
        against the summaries of reference candidates to pick the correct ID 
        or declare it a 'NEW_ENTITY'.
        """
        candidates = row.get("anchoring_candidates", [])
        if not candidates:
            return {"choice": "NEW_ENTITY"}

        candidates_text = self._format_anchoring_candidates(candidates)
        full_context = " ".join(set(row.description)) if isinstance(row.description, list) else str(row.description)
        entity_context = full_context.replace("\n", " ").strip()[:250]

        try:
            # Standardized JSON response for direct programmatic integration
            return await self.llm_service.ask_json(
                system_prompt=ANCHORING_RESOLUTION_SYSTEM_PROMPT,
                user_prompt=ANCHORING_RESOLUTION_USER_PROMPT.format(
                    entity_title=row.title,
                    entity_type=row.type,
                    entity_context=entity_context,
                    candidates_text=candidates_text
                )
            )
        except Exception as e:
            logger.error(f"❌ LLMResolver anchoring error ({row.title}): {e}")
            return {"choice": "NEW_ENTITY"} # Fallback as a new entity 



    def _create_hybrid_clusters(self, df: pd.DataFrame) -> list:
        """
        Groups entities into 'potential duplicate clusters' using a graph-based approach.
        
        Algorithm:
        1. Nodes = Entities.
        2. Edges = Created if two entities share semantic similarity (TF-IDF cosine > 0.3), 
           structural similarity (Levenshtein > 0.7), or name containment.
        3. Clusters = Connected components of the resulting graph.
        
        This reduces the number of LLM calls by only comparing entities that have 
        a base
        """
        if len(df) <= 1: return [df]

        df = df.copy()
        df["norm_name"] = df["title"].apply(normalize_entity_name)
        
        # Combine name and description for TF-IDF vectorization
        texts = df.apply(lambda x: f"{x['title']} {' '.join(x['description'])}", axis=1).tolist()
        vectorizer = TfidfVectorizer(stop_words='english') 
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix)

        G = nx.Graph()
        G.add_nodes_from(range(len(df)))

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Similarity criteria (Layered approach)
                is_semantic = cosine_sim[i][j] >= 0.3
                name_i, name_j = df.iloc[i]["norm_name"], df.iloc[j]["norm_name"]
                is_contained = (name_i in name_j) or (name_j in name_i) if (len(name_i) > 2 and len(name_j) > 2) else False
                is_fuzzy = similarity(name_i, name_j) >= 0.7

                if is_semantic or is_contained or is_fuzzy:
                    G.add_edge(i, j)

        # Return list of DataFrames, each representing a connected component (cluster)
        clusters = list(nx.connected_components(G))
        
        logger.debug(f"📊 Hybrid clustering created {len(clusters)} groups from {len(df)} nodes.")
        return [df.iloc[list(c)] for c in clusters]

    
    def _format_anchoring_candidates(self, candidates: list) -> str:
        """Formats the list of reference candidates for the LLM prompt."""
        return "\n".join([f"- ID: {c.get('ID', 'UNKNOWN')} | Name: {c.get('CANONICAL_NAME', 'UNKNOWN')} | Summary: {c.get('CORE_SUMMARY', 'No summary.')}" for c in candidates])