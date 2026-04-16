import asyncio
import pandas as pd
import logging
from typing import List, Any, Tuple, Dict

from app.indexing.operations.graph.utils import filter_orphan_relationships
from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
from app.indexing.operations.graph.summarize_manager import SummarizeManager

from app.services.llm.parser import LLMParser
from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine

from app.indexing.operations.graph.store_manager import GraphStoreManager

import logging
logger = logging.getLogger(__name__)

class GraphService:
    """
    Orchestrator for the GraphRAG indexing pipeline.
    
    This service coordinates the lifecycle of knowledge graph construction:
    1. Distributed Extraction: Parallel LLM calls to extract entities/relationships.
    2. Entity Resolution: Merging duplicates using deterministic and LLM-based logic.
    3. Relationship Mapping: Re-anchoring relations to resolved entity IDs.
    4. Summarization: Consolidating descriptions for final graph storage.
    """
    def __init__(
        self, 
        extractor: EntityAndRelationExtractor, 
        summarizer: SummarizeManager, 
        parser: LLMParser, 
        resolution_engine: EntityResolutionEngine,
        store_manager: GraphStoreManager
    ):
        self.extractor = extractor
        self.summarizer = summarizer
        self.parser = parser
        self.resolution_engine = resolution_engine
        self.store_manager = store_manager

    async def run_pipeline(
        self, 
        text_units: List[Any], 
        domain_context: str,
        persist: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the end-to-end graph construction pipeline.
        
        Args:
            text_units: List of TextUnits to process.
            domain_context: Global document identity used to ground the LLM.
            
        Returns:
            Tuple containing the final Entities and Relationships DataFrames.
        """
        logger.info(f"🕸️ Starting graph pipeline for {len(text_units)} units.")

        # 1. EXTRACTION & PARSING
        logger.info("📡 Phase 1: Distributed extraction (Parallel LLM calls)...")
        tasks = [self.extractor(u.text, domain_context) for u in text_units]
        raw_results = await asyncio.gather(*tasks)
        source_ids = [u.id for u in text_units]
        
        entities_df, relationships_df = self.parser.to_dataframes(raw_results, source_ids)

        if entities_df.empty:
            logger.warning("⚠️ Phase 1 resulted in 0 entities. Pipeline aborted.")
            return entities_df, relationships_df

        logger.info(f"📊 Extraction complete: {len(entities_df)} raw entities, {len(relationships_df)} raw relations.")

        # 2. ENTITY RESOLUTION (The Magic happens here)
        logger.info("🧠 Phase 2: Entity Resolution & Fusion...")
        # The Engine handles Core, LLM, Transitive Mapping and Row Fusion
        entities_df, global_mapping = await self.resolution_engine.run(entities_df)

        # Frequency update post-fusion (count unique source_ids per resolved entity)
        entities_df["frequency"] = entities_df["source_id"].apply(len)
        logger.info(f"✅ Resolution complete. Entities reduced to {len(entities_df)} unique nodes.")

        # 3. RELATIONSHIPS RE-MAPPING
        if not relationships_df.empty:
            logger.info("🔗 Phase 3: Re-mapping relationships to resolved entities...")
            relationships_df = self._process_relationships(relationships_df, global_mapping)
        else:
            logger.debug("No relationships found to process.")

        # 4. ORPHAN FILTERING
        # Ensure every source/target in relationships still exists in the resolved entities list
        pre_filter_count = len(relationships_df)
        relationships_df = filter_orphan_relationships(relationships_df, entities_df)
        logger.debug(f"🧹 Filtered {pre_filter_count - len(relationships_df)} orphan relationships.")

        # 5. SUMMARIZATION
        logger.info("📝 Phase 4: Summarizing consolidated descriptions...")
        entities_df, relationships_df = await self.summarizer.summarize_all(
            entities_df=entities_df,
            relationships_df=relationships_df
        )

        # 6. PERSISTENCE 
        if persist:
            logger.info("💾 Phase 5: Persisting graph to Neo4j...")
            try:
                await self.store_manager.save_graph(entities_df, relationships_df)
            except Exception as e:
                logger.error(f"❌ Failed to persist graph to Neo4j: {e}")
                # We don't raise here to allow the method to return the DFs

        logger.info("🏁 Graph pipeline finished successfully.")
        return entities_df, relationships_df

    def _process_relationships(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Cleans and aggregates relationships post-resolution.
        
        1. Applies transitive mapping (A -> C) to source and target columns.
        2. Removes self-loops (A -> A).
        3. Aggregates multi-edges into a single edge with weight and combined descriptions.
        """

        if mapping:
            # Replace old names with resolved representative names
            df["source"] = df["source"].replace(mapping)
            df["target"] = df["target"].replace(mapping)

        # Check for self-loops
        loops = df[df["source"] == df["target"]]
        if not loops.empty:
            logger.info(f"🔄 [DEBUG] Removing {len(loops)} self-loops created by entity fusion.")

        df = df[df["source"] != df["target"]]
        
        # Aggregate redundant relations (e.g., same relation mentioned in different chapters)
        return (
            df.groupby(["source", "target"], sort=False)
            .agg({
                "description": list, 
                "weight": "sum", 
                "source_id": lambda x: list(set([
                    item for sublist in (x if isinstance(x.iloc[0], list) else [x]) 
                    for item in sublist
                ])) 
            })
            .reset_index()
        )