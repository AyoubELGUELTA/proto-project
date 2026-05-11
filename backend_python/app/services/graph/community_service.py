import logging
import pandas as pd
from typing import Dict, Any, List
from app.infrastructure.neo4j.client import Neo4jClient

logger = logging.getLogger(__name__)

class CommunityService:
    """
    Service responsible for Neo4j read/write operations related to graph communities.

    This service acts as the data access layer for community detection and reporting,
    bridging the gap between Neo4j persistence and graph analysis libraries.

    Attributes:
        client (Neo4jClient): The infrastructure client used to execute Cypher queries.
    """

    def __init__(self, client: Neo4jClient):
        """
        Initializes the CommunityService with a Neo4j client.

        Args:
            client (Neo4jClient): An instance of the Neo4j infrastructure client.
        """
        self.client = client

    async def get_relationships_for_clustering(self) -> pd.DataFrame:
        """
        Extracts all relationships from the graph to feed the clustering algorithm.

        Retrieves source and target entity IDs along with their relationship weights.
        This data is essential for algorithms like Leiden or Louvain.

        Returns:
            pd.DataFrame: A DataFrame containing ['source', 'target', 'weight'] columns.
                Returns an empty DataFrame if no relationships are found.

        Raises:
            Exception: If the Cypher query execution fails.
        """
        query = """
        MATCH (s:Entity)-[r:RELATED]->(t:Entity)
        RETURN s.id AS source, t.id AS target, r.weight AS weight
        """
        try:
            records = await self.client.execute_query(query)
            
            if not records:
                logger.warning("⚠️ No relationships found in Neo4j for clustering.")
                return pd.DataFrame(columns=['source', 'target', 'weight'])
            
            df = pd.DataFrame(records)
            
            # Ensure weights are numeric to prevent clustering failures
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
            
            logger.info(f"📊 Extracted {len(df)} relationships for community detection.")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to extract relationships for clustering: {e}")
            raise

    async def get_all_entity_titles(self) -> Dict[str, str]:
        """
        Retrieves a mapping of Entity IDs to their Titles.

        Used during the community summarization phase to provide human-readable 
        context to the LLM when describing clusters.

        Returns:
            Dict[str, str]: A dictionary where keys are entity IDs and values are titles.

        Raises:
            Exception: If the Cypher query execution fails.
        """
        query = "MATCH (e:Entity) RETURN e.id AS id, e.title AS title"
        try:
            records = await self.client.execute_query(query)
            mapping = {r['id']: r['title'] for r in records}
            
            logger.debug(f"📑 Retrieved title mapping for {len(mapping)} entities.")
            return mapping
        except Exception as e:
            logger.error(f"❌ Failed to retrieve entity titles: {e}")
            raise