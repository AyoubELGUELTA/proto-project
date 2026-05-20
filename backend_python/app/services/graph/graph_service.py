import asyncio
import pandas as pd
import logging
from typing import List, Any, Tuple, Dict

from app.core.data_model.base import slugify_entity
from app.core.data_model.text_units import TextUnit

from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
from app.indexing.operations.graph.summarize_manager import SummarizeManager
from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine
from app.indexing.operations.graph.store_manager import GraphStoreManager

from backend_python.app.services.graph.community_service import CommunityService

from app.services.llm.parser import LLMParser

from app.indexing.workflows.create_communities import run_create_communities_workflow

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
    5. Persistence: We construct the graph with the provided data.
    """
    def __init__(
        self, 
        extractor: EntityAndRelationExtractor, 
        summarizer: SummarizeManager, 
        parser: LLMParser, 
        resolution_engine: EntityResolutionEngine,
        store_manager: GraphStoreManager,
        community_service: CommunityService
    ):
        self.extractor = extractor
        self.summarizer = summarizer
        self.parser = parser
        self.resolution_engine = resolution_engine
        self.store_manager = store_manager
        self.community_service = community_service

    async def run_pipeline(
        self, 
        text_units: List[TextUnit], 
        domain_context: str,
        persist: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the end-to-end knowledge graph construction pipeline.
        
        This orchestrator handles extraction, entity resolution, relationship 
        re-mapping, summarization, and persistence. It ensures data consistency 
        by bridging initial LLM extractions (slugs) with final resolved UUIDs.
        
        Args:
            text_units (List[TextUnit]): The text chunks to process.
            domain_context (str): Global context to ground the LLM extractions.
            persist (bool): If True, saves the final dataframes to the graph database.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The processed (Entities DF, Relationships DF).
        """
        logger.info(f"🕸️ Starting graph pipeline for {len(text_units)} units.")

        # --- Phase 1: EXTRACTION ---
        logger.info("📡 Phase 1: Distributed extraction via LLM...")
        tasks = [self.extractor(u.text, domain_context) for u in text_units]
        raw_results = await asyncio.gather(*tasks)

        source_ids = [u.id for u in text_units]
        entities_df, relationships_df = self.parser.to_dataframes(raw_results, source_ids)

        if entities_df.empty:
            logger.warning("⚠️ Phase 1 yielded no entities. Aborting pipeline.")
            return entities_df, relationships_df

        logger.info(f"📊 Extracted: {len(entities_df)} raw entities, {len(relationships_df)} raw relations.\n")


        # --- Phase 2: ENTITY RESOLUTION ---
        entities_df["slug"] = entities_df["title"].apply(slugify_entity)
        initial_slug_to_id = dict(zip(entities_df["slug"], entities_df["id"]))
        
        # SAUVEGARDE STRICTE DE L'ÉTAT INITIAL
        # On garde une copie immuable pour aller chercher les vrais noms initiaux
        pre_resolution_df = entities_df.copy()

        logger.info("🧠 Phase 2: Entity Resolution & Fusion...")
        entities_df, global_mapping = await self.resolution_engine.run(entities_df)
        
        # LOGS DE DIAGNOSTIC AMONT LISIBLES ET BLINDÉS
        logger.info(f"🔍 [DIAGNOSTIC] global_mapping size: {len(global_mapping)}")
        logger.info(f"🔍 [DIAGNOSTIC] Redirections (Old Title ==> New Title):")
        
        gm_samples = list(global_mapping.items())[:35]
        for old_id, new_id in gm_samples:
            old_match = pre_resolution_df[pre_resolution_df["id"] == old_id]
            old_name = old_match.iloc[0]["title"] if not old_match.empty else "Unknown_Old_ID"
            
            new_match = entities_df[entities_df["id"] == new_id]
            new_name = new_match.iloc[0]["title"] if not new_match.empty else "Unknown_New_ID"
            
            logger.info(f"  {old_name} ({old_id})  ==>  {new_name} ({new_id})")

        # Update frequency
        entities_df["frequency"] = entities_df["source_ids"].apply(len)
        logger.info(f"✅ Resolution complete: {len(entities_df)} unique entity nodes.\n")


        # --- Phase 3: RELATIONSHIP RE-MAPPING ---
        if not relationships_df.empty:
            logger.info("🔗 Phase 3: Re-mapping relations...")

            final_slug_map = {}
            for slug, init_id in initial_slug_to_id.items():
                final_id = global_mapping.get(init_id, init_id)
                final_slug_map[slug] = final_id

            relationships_df = self._process_relationships(
                relationships_df, 
                final_slug_map
            )
        else:
            logger.info("ℹ️ No relationships found to process.\n")


        # --- Phase 4: SUMMARIZATION ---
        logger.info("📝 Phase 4: Summarizing consolidated descriptions...")
        entities_df, relationships_df = await self.summarizer.summarize_all(
            entities_df=entities_df,
            relationships_df=relationships_df
        )
        logger.info("✅ Summarization complete.\n")


        # --- Phase 5: PERSISTENCE (Existing) ---
        if persist:
            logger.info("💾 Phase 5: Persisting graph to database...")
            try:
                await self.store_manager.save_graph(entities_df, relationships_df)
                logger.info("✅ Graph successfully saved to Neo4j.")
                
                # --- Phase 6: COMMUNITY DETECTION  --- # We trigger this ONLY if persistence was successful
                logger.info("🏘️ Phase 6: Building community hierarchy...")
                
                await run_create_communities_workflow(self.community_service)
                
            except Exception as e:
                logger.error(f"❌ Failed during graph persistence/clustering: {e}")

        logger.info("🏁 Graph pipeline finished successfully.")
        return entities_df, relationships_df
    
    def _process_relationships(
        self, 
        rels_df: pd.DataFrame, 
        slug_to_id: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Maps raw extracted relationships to their final resolved entity IDs.
        
        This method uses a direct mapping:
        Slug -> Final ID (as resolved by the engine)
        
        Args:
            rels_df (pd.DataFrame): The raw relationships dataframe.
            slug_to_id (Dict[str, str]): Final mapping of slugs to consolidated IDs.
            
        Returns:
            pd.DataFrame: Cleaned, re-mapped, and aggregated relationships.
        """
        if rels_df.empty:
            return rels_df

        # Step 1: Map raw slugs directly to final consolidated IDs
        rels_df["source_id"] = rels_df["source_slug"].map(slug_to_id)
        rels_df["target_id"] = rels_df["target_slug"].map(slug_to_id)

        # Step 2: Drop hard orphans (slugs that never existed in final entities_df)
        before_drop = len(rels_df)
        rels_df = rels_df.dropna(subset=["source_id", "target_id"])
        
        dropped = before_drop - len(rels_df)
        if dropped > 0:
            logger.warning(f"🧹 {dropped} relationships dropped due to missing entity IDs (hallucinations).")

        # Step 3: Remove self-loops created by entity fusion (e.g., Muhammad -> Prophet becomes Muhammad -> Muhammad)
        rels_df = rels_df[rels_df["source_id"] != rels_df["target_id"]]

        # Step 4: Aggregate duplicate relationships
        return (
            rels_df.groupby(["source_id", "target_id"], sort=False)
            .agg({
                "source_slug": "first",
                "target_slug": "first",
                "description": lambda x: " | ".join(set(filter(None, x))),
                "weight": "sum",
                "source_ids": lambda x: list(set([i for sub in x for i in sub if isinstance(sub, list)])),
                "rank": "max",          # Retain the highest importance rank
                "attributes": "first"   # Retain the first encountered attributes dict
            })
            .reset_index()
        )