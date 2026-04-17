from app.indexing.operations.entity_resolution.identity_tracker import IdentityTracker
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver

from typing import Tuple

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

    async def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Executes the full entity resolution pipeline on an extraction batch.
        
        This method follows a 'progressive refinement' strategy: it clears the obvious 
        duplicates using deterministic rules first to minimize expensive LLM calls.
        
        Args:
            df: The raw extracted entities DataFrame.
            
        Returns:
            - final_df: A deduplicated DataFrame where each row is a unique entity.
            - final_map: The complete identity mapping used for relationship remapping.
        """
        if df.empty:
            return df, {}

        initial_count = len(df)
        logger.info(f"🚀 Starting Entity Resolution Engine on {initial_count} raw entities.")
        tracker = IdentityTracker()
        
        # --- 1. CORE RESOLUTION (Phonetic & Exact) ---
        df, core_mappings = self.core.resolve(df)
        for old, new in core_mappings.items(): 
            tracker.add_mapping(old, new)
        
        # --- 2. SEMANTIC RESOLUTION (LLM) ---
        try:
            # Note: llm_resolve now uses 'category' internally thanks to our refactor

            df, llm_mappings = await self.llm.llm_resolve(df)
            for old, new in llm_mappings.items(): 
                tracker.add_mapping(old, new)
        except Exception as e:
            logger.error(f"❌ LLM Resolution Error: {e}")

        # --- 3. ENCYCLOPEDIA ANCHORING ---
        # If a canonical_id was found, it becomes the ultimate ground truth
        mask = df["canonical_id"].notna()
        if mask.any():
            for _, row in df[mask].iterrows():
                # We map the current title (slug) to the Encyclopedia ID
                tracker.add_mapping(row["title"], row["canonical_id"])

        # --- 4. FINAL TRANSITIVE RESOLUTION ---
        # Flattens chains: A -> B -> C becomes A -> C
        final_map = tracker.resolve()

        # --- 5. PHYSICAL UPDATE ---
        # We apply the map. Titles might now become Encyclopedia IDs
        df["title"] = df["title"].replace(final_map)
        
        # --- 6. PHYSICAL AGGREGATION ---
        # Consolidate rows that share the same final identity
        final_df = self._aggregate_entities(df)

        # Integrity check
        unique_identities = set(final_df["title"])
        errors = [v for v in final_map.values() if v not in unique_identities]
        if errors:
            logger.warning(f"⚠️ {len(set(errors))} mapping targets missing from final nodes.")

        logger.info(f"✅ Resolution Complete: {initial_count} -> {len(final_df)} entities.")        
        return final_df, final_map
    
    
    def _aggregate_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physically merges duplicate rows into unified entity records.
        
        This method performs a GroupBy operation on ['title', 'type']. 
        The 'type' is included to prevent accidental merging of homonyms 
        of different natures (e.g., a Person and a Location with the same name).
        
        Aggregation Rules:
        - description: List concatenation (prepared for the SummarizeManager).
        - source_id: Unique list union to track provenance.
        - frequency: Sum of occurrences.
        - canonical_id: Persistence of the anchored ID.
        """
        def merge_descriptions(series):
            # Cleanly merge description strings, avoiding empty values
            return " | ".join(set(filter(None, series.astype(str))))

        def merge_sources(series):
            # Flatten lists of source_ids and keep unique values
            combined = []
            for s_list in series:
                if isinstance(s_list, list): combined.extend(s_list)
            return list(set(combined))

        agg_rules = {
            "description": merge_descriptions,
            "source_ids": merge_sources, 
            "frequency": "sum",
            "category": "first",         # Category is stable for a given title/ID
            "canonical_id": "first"
        }
        
        # We group by 'title' and 'type'. 
        # Note: If title is an Encyclopedia ID, type is already standardized.
        return df.groupby(["title", "type"], sort=False).agg(agg_rules).reset_index()
    
















