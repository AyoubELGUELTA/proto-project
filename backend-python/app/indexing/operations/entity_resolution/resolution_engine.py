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
        
        # --- 1. RESOLUTION (Core + LLM) ---
        # Collect all redirections suggested by deterministic and semantic algorithms
        df, core_mappings = self.core.resolve(df)
        for old, new in core_mappings.items(): 
            tracker.add_mapping(old, new)
        
        try:
            df, llm_mappings = await self.llm.resolve_complex_cases(df)
            for old, new in llm_mappings.items(): 
                tracker.add_mapping(old, new)
        except Exception as e:
            logger.error(f"❌ LLM Resolution Error: {e}")

        # --- 2. ENCYCLOPEDIA ANCHORING ---
        # If a canonical_id was found, we inject it as the final hop in the redirection chain.
        # This ensures that even resolved names point to the official Encyclopedia ID.

        mask = df["canonical_id"].notna()
        if mask.any():
            logger.debug(f"⚓ Injecting {mask.sum()} Encyclopedia anchors into tracker.")
            for _, row in df[mask].iterrows():
                tracker.add_mapping(row["title"], row["canonical_id"])

        # --- 3. FINAL TRANSITIVE RESOLUTION ---
        # The tracker flattens chains (e.g., 'A' -> 'B' -> 'C' implies 'A' -> 'C')
        final_map = tracker.resolve()

        # --- 4. PHYSICAL UPDATE & AGGREGATION ---
        # Update titles in the DataFrame using the flattened mapping
        df["title"] = df["title"].replace(final_map)
        
        # Consolidate rows that now share the same title/ID (post-anchoring duplicates)
        final_df = self._aggregate_entities(df)

        # --- 5. VALIDATION ---
        # Verify that all mapping targets actually exist in the final nodes list
        unique_titles = set(final_df["title"])
        errors = [v for v in final_map.values() if v not in unique_titles]
        if errors:
            logger.warning(f"⚠️ {len(set(errors))} mapping targets missing from final nodes (Integrity risk).")

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
        if df.empty:
            return df

        # Ensure descriptions are lists to support the 'sum' aggregation (list concatenation)
        if not isinstance(df.iloc[0]["description"], list):
            df["description"] = df["description"].apply(lambda d: [d] if isinstance(d, str) else d)

        agg_rules = {
            "description": "sum",  # Merges lists of strings: [d1] + [d2] -> [d1, d2]
            "source_id": "sum",    # Merges provenance IDs
            "frequency": "sum",
            "canonical_id": "first"
        }
        
        # Grouping by both title and type preserves semantic distinctions
        return df.groupby(["title", "type"], sort=False).agg(agg_rules).reset_index()
    
















