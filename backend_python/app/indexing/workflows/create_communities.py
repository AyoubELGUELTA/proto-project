import logging
import pandas as pd
from typing import Optional
from app.services.graph.community_service import CommunityService
from app.indexing.operations.communities.cluster_graph import run_clustering

logger = logging.getLogger(__name__)

async def run_create_communities_workflow(community_service: CommunityService) -> Optional[pd.DataFrame]:
    """
    Orchestrates the community detection workflow.

    This workflow follows the GraphRAG (Microsoft-inspired) approach:
    1. Fetches the global relationship state from Neo4j.
    2. Performs hierarchical clustering using the Leiden algorithm (via Graspologic).
    3. Prepares the resulting clusters for subsequent summarization and persistence.

    Args:
        community_service (CommunityService): The service used to interact with 
            Neo4j community data.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the hierarchical cluster 
            mapping if successful, None otherwise.
    """
    logger.info("🚀 Launching Community Creation Workflow...")

    # --- Step 1: Data Extraction ---
    # We pull the resolved graph state to ensure we cluster cleaned entities
    relationships_df = await community_service.get_relationships_for_clustering()
    
    if relationships_df.empty:
        logger.error("❌ Workflow aborted: Insufficient relationship data for clustering.")
        return None

    # --- Step 2: Community Detection (Compute) ---
    # We execute the Hierarchical Leiden algorithm
    logger.info(f"🧠 Computing Leiden clusters for {len(relationships_df)} relationships...")
    
    try:
        # max_cluster_size is a key hyperparameter for LLM context window management later
        clusters_df = run_clustering(relationships_df, max_cluster_size=15)#TODO centralisé max_cluster_size dans les configs..
        
        if clusters_df.empty:
            logger.warning("⚠️ Clustering engine returned an empty result.")
            return None

        logger.info(
            f"✅ Clustering complete: {clusters_df['community'].nunique()} "
            f"communities detected across hierarchy levels."
        )
        
        # --- Step 3: Metadata Enrichment (Future) ---
        # TODO: Integrate title mapping and LLM Summarization here
        
        return clusters_df

    except Exception as e:
        logger.error(f"❌ Failed to execute community detection: {e}")
        return None