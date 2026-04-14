from app.indexing.operations.entity_resolution.identity_tracker import IdentityTracker
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver

from typing import Tuple
import pandas as pd


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

        # The tracker maintains the global 'who is who' state during the run
        tracker = IdentityTracker()
        
        # 1. CORE RESOLUTION PHASE
        # Uses phonetic similarity and exact Encyclopedia lookups.
        df, core_mappings = self.core.resolve(df)
        for old, new in core_mappings.items():
            tracker.add_mapping(old, new)

        # 2. LLM RESOLUTION PHASE
        # Tackles complex cases (anchoring doubts or semantic variants).
        df, llm_mappings = await self.llm.resolve_complex_cases(df)
        for old, new in llm_mappings.items():
            tracker.add_mapping(old, new)

        # 3. TRANSITIVE MAPPING GENERATION
        # Flattens all chains (e.g., A -> B -> ID_1 becomes A -> ID_1).
        final_map = tracker.resolve()

        # 4. DATAFRAME HARMONIZATION
        # Apply the final names/IDs to the title column.
        df["title"] = df["title"].replace(final_map)
        
        # SAFETY INVARIANT: If an entity is anchored to a Sira Encyclopedia ID,
        # its 'title' MUST become that ID to ensure global graph consistency.
        mask = df["canonical_id"].notna()
        df.loc[mask, "title"] = df.loc[mask, "canonical_id"]

        # 5. FINAL PHYSICAL AGGREGATION
        # Merges rows that now share the same title and type.
        final_df = self._aggregate_entities(df)

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