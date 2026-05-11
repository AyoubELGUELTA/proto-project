from app.indexing.operations.entity_resolution.identity_tracker import IdentityTracker
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.core.data_model.entity import EntityModel

from typing import Tuple, List, Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class EntityResolutionEngine:
    """
    The orchestrator for the entire entity resolution and deduplication pipeline.
    
    This engine executes a multi-stage process to ensure graph integrity:
    1. Deterministic Resolution (Core): Fast, phonetic-based clustering and exact matching.
    2. Semantic Resolution (LLM): Context-aware anchoring and complex duplicate merging.
    3. Identity Consolidation: Resolving transitive chains of renames.
    4. Physical Merging: Aggregating descriptions and metadata into a single record per identity.
    """
    def __init__(self, core_resolver: CoreResolver, llm_resolver: LLMResolver):
        """
        Initializes the engine with its two specialized resolution layers.
        
        Args:
            core_resolver: Handles algorithmic and encyclopedia-based matching.
            llm_resolver: Handles semantic clusters and anchoring ambiguities via AI.
        """
        self.core = core_resolver
        self.llm = llm_resolver

    async def run(self, entities: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Executes the full entity resolution pipeline on an extraction batch.

        This method follows a progressive refinement strategy:
        it clears obvious duplicates using deterministic rules first to minimize expensive LLM calls.

        Args:
            entities:
                DataFrame containing the raw extracted EntityModel objects.

        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]:
                - final_entities:
                    Deduplicated entities where each node is unique.
                - final_map:
                    Complete identity mapping used for relationship remapping.
        """
        if entities.empty:
            logger.warning("⚠️ No entities received in the engine. Pipeline skip.")
            return entities, {}

        initial_count = len(entities)
        logger.info(f"🚀 Entity Resolution Start: {initial_count} raw entities.")
        tracker = IdentityTracker()
    
        # --- 1. CORE RESOLUTION (Phonetic & Exact) ---
        try:
            id_to_title = dict(zip(entities['id'], entities['title']))
            entities, core_mappings = await self.core.resolve(entities)
            for old_id, new_id in core_mappings.items(): 
                tracker.add_mapping(old_id, new_id)
                logger.info(f" [CORE MERGE] {id_to_title.get(old_id)} -> {id_to_title.get(new_id)}")
                
            logger.info(f"🔹 Core: {len(core_mappings)} deterministic mappings created.")
        except Exception as e:
            logger.error(f"❌ Core Resolution Error: {e}")
        
        # --- 2. SEMANTIC RESOLUTION (LLM) ---
        try:
            id_to_title = dict(zip(entities['id'], entities['title']))
            entities, llm_mappings = await self.llm.llm_resolve(entities)
            for old_id, new_id in llm_mappings.items(): 
                tracker.add_mapping(old_id, new_id)
                logger.info(f" 🧠 [LLM MERGE] {id_to_title.get(old_id)} -> {id_to_title.get(new_id)}")

            logger.info(f"🔹 LLM: {len(llm_mappings)} semantic mappings created.")
        except Exception as e:
            logger.error(f"❌ LLM Resolution Error: {e}")

        # --- 3. ENCYCLOPEDIA ANCHORING ---
        anchoring_count = 0
        for row in entities.itertuples():
            if hasattr(row, 'canonical_id') and row.canonical_id:
                tracker.add_mapping(row.id, row.canonical_id)
                anchoring_count += 1
        if anchoring_count > 0:
            logger.info(f"⚓ Anchoring: {anchoring_count} entities linked to Encyclopedia UUIDs.")

        # --- 4. FINAL TRANSITIVE RESOLUTION ---
        # Get the explicit chain of redirects (A -> B -> C becomes A -> C)
        final_map = tracker.resolve()

        # --- 5. PHYSICAL UPDATE ---
        # We apply the explicit mapping to the 'id' column
        if final_map:
            entities['id'] = entities['id'].map(final_map).fillna(entities['id'])

        # --- 6. PHYSICAL AGGREGATION ---
        # Consolidate objects sharing the same ID
        final_entities = self._aggregate_entities(entities)

        diff = initial_count - len(final_entities)
        logger.info(f"🔗 Final Map: {len(final_map)} total redirections.")
        logger.info(f"✅ Resolution Complete: {initial_count} -> {len(final_entities)} (Merged {diff} duplicates).")

        return final_entities, final_map
    
    def _aggregate_entities(self, entities_df: pd.DataFrame) -> pd.DataFrame:       
        """
        Physically merges duplicate EntityModel objects into unified records.
        
        This method performs a GroupBy operation using Pandas to handle the 
        complex logic of merging source lists and descriptions efficiently.
        """
        if entities_df.empty:
            return []

        def merge_descriptions(series):
            # Cleanly merge description strings, avoiding empty values and duplicates
            return " | ".join(set(filter(None, series.astype(str))))

        def merge_sources(series):
            # Flatten lists of source_ids and keep unique values
            combined = []
            for s_list in series:
                if isinstance(s_list, list): combined.extend(s_list)
            return list(set(combined))

        def merge_attributes(series):
            # Merge dictionary attributes (last one wins for overlapping keys)
            final_attr = {}
            for attr_dict in series:
                if isinstance(attr_dict, dict):
                    final_attr.update(attr_dict)
            return final_attr

        def merge_communities(series):
            # Union of community ID lists
            combined = set()
            for c_list in series:
                if isinstance(c_list, list): combined.update(c_list)
            return list(combined)

        agg_rules = {
            "title": "first",
            "description": merge_descriptions,
            "source_ids": merge_sources, 
            "frequency": "sum",
            "rank": "max",
            "category": "first",
            "canonical_id": "first",
            "review_status": "first",
            "attributes": merge_attributes,
            "community_ids": merge_communities
        }
        
        return entities_df.groupby(["id"], sort=False).agg(agg_rules).reset_index()