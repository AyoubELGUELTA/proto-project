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

    async def run(self, entities: List[EntityModel]) -> Tuple[List[EntityModel], Dict[str, str]]:
        """
        Executes the full entity resolution pipeline on an extraction batch.
        
        This method follows a 'progressive refinement' strategy: it clears the obvious 
        duplicates using deterministic rules first to minimize expensive LLM calls.
        
        Args:
            entities: The list of raw extracted EntityModel objects.
            
        Returns:
            - final_entities: A deduplicated list of EntityModel where each entry is unique.
            - final_map: The complete identity mapping used for relationship remapping.
        """
        if not entities:
            return [], {}

        initial_count = len(entities)
        logger.info(f"🚀 Starting Entity Resolution Engine on {initial_count} raw entities.")
        tracker = IdentityTracker()
        
        # --- 1. CORE RESOLUTION (Phonetic & Exact) ---
        # We process the models through the deterministic layer
        entities, core_mappings = await self.core.resolve(entities)
        for old_id, new_id in core_mappings.items(): 
            tracker.add_mapping(old_id, new_id)
        
        # --- 2. SEMANTIC RESOLUTION (LLM) ---
        try:
            # The LLM layer settles remaining ambiguities and clusters orphans
            entities, llm_mappings = await self.llm.llm_resolve(entities)
            for old_id, new_id in llm_mappings.items(): 
                tracker.add_mapping(old_id, new_id)
        except Exception as e:
            logger.error(f"❌ LLM Resolution Error: {e}")

        # --- 3. ENCYCLOPEDIA ANCHORING ---
        # If a canonical_id was found during any phase, it becomes the ultimate ground truth
        for entity in entities:
            if entity.canonical_id:
                # We map the current title (or any previous version) to the Encyclopedia ID
                tracker.add_mapping(entity.id, entity.canonical_id)

        # --- 4. FINAL TRANSITIVE RESOLUTION ---
        # Flattens chains: A -> B -> C becomes A -> C
        final_map = tracker.resolve()

        # --- 5. PHYSICAL UPDATE ---
        # We update titles based on the final mapping. Titles might now become Encyclopedia IDs.
        for entity in entities:
            if entity.id in final_map:
                entity.id = final_map[entity.id]
        
        # --- 6. PHYSICAL AGGREGATION ---
        # Consolidate objects that now share the same title/identity
        # We convert to DataFrame temporarily for high-performance grouping
        final_entities = self._aggregate_entities(entities)

        logger.info(f"✅ Resolution Complete: {initial_count} -> {len(final_entities)} entities.")        
        return final_entities, final_map
    
    def _aggregate_entities(self, entities: List[EntityModel]) -> List[EntityModel]:
        """
        Physically merges duplicate EntityModel objects into unified records.
        
        This method performs a GroupBy operation using Pandas to handle the 
        complex logic of merging source lists and descriptions efficiently.
        """
        if not entities:
            return []

        # Convert models to DataFrame for aggregation
        df = pd.DataFrame([e.model_dump() for e in entities])

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
        
        # Group by 'id', it ensure unicity
        grouped = df.groupby(["id"], sort=False).agg(agg_rules).reset_index()

        # Convert back to EntityModel objects
        # pd.notna(v) check is used to ensure NaN becomes None for Pydantic
        return [
            EntityModel(**{k: (v if pd.notna(v) else None) for k, v in row.items()}) 
            for _, row in grouped.iterrows()
        ]