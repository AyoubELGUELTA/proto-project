import logging
import pandas as pd
import json
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
                return pd.DataFrame(columns=['source_', 'target', 'weight'])
            
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


    async def save_community_assignments(self, clusters_df: pd.DataFrame):
        """
        Persists community assignments into Neo4j.

        Creates :Community nodes and links them to existing :Entity nodes.
        Also establishes the hierarchical parent-child relationships between communities.

        Args:
            clusters_df (pd.DataFrame): DataFrame with ['level', 'community', 'parent', 'node'].
        """
        if clusters_df.empty:
            logger.warning("⚠️ No clusters to save.")
            return

        # Prepare a unique ID for communities across levels: e.g., "level_1_id_42"
        clusters_df['community_uid'] = clusters_df.apply(
            lambda x: f"level_{x['level']}_id_{x['community']}", axis=1
        )
        clusters_df['parent_uid'] = clusters_df.apply(
            lambda x: f"level_{int(x['level'])-1}_id_{x['parent']}" if x['parent'] != -1 else None, 
            axis=1
        )

        # 1. Create Community Nodes and Link Entities
        # We use UNWIND for bulk processing (highly efficient in Neo4j)
        entity_link_query = """
        UNWIND $data AS row
        MERGE (c:Community {id: row.community_uid})
        SET c.level = row.level,
            c.id_in_level = row.community,
            c.title = "Community " + row.community_uid
        
        WITH c, row
        MATCH (e:Entity {id: row.node})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """

        # 2. Build the Hierarchy (Parent-Child)
        hierarchy_query = """
        UNWIND $data AS row
        WITH row WHERE row.parent_uid IS NOT NULL
        MATCH (child:Community {id: row.community_uid})
        MATCH (parent:Community {id: row.parent_uid})
        MERGE (child)-[:CHILD_OF]->(parent)
        """

        try:
            data_payload = clusters_df.to_dict(orient='records')
            
            # Step 1: Link Entities
            await self.client.execute_query(entity_link_query, {"data": data_payload})
            # Step 2: Set Hierarchy
            await self.client.execute_query(hierarchy_query, {"data": data_payload})
            
            logger.info(f"💾 Successfully persisted {len(clusters_df)} community assignments to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Failed to persist communities: {e}")
            raise
    
    async def get_all_community_ids(self) -> List[str]:
        """
        Retrieves the unique IDs of all communities currently stored in Neo4j.
        Used to orchestrate the pipeline loop over communities.
        """
        query = "MATCH (c:Community) RETURN c.id AS id"
        try:
            records = await self.client.execute_query(query)
            return [r['id'] for r in records]
        except Exception as e:
            logger.error(f"❌ Failed to retrieve community IDs: {e}")
            raise

    async def get_community_context(self, community_id: str) -> Dict[str, Any]:
        """
        Extracts all raw components (Entities and internal Relationships) 
        of a community to form the baseline text context for LLM Summarization.
        
        Args:
            community_id (str): The unique ID of the target community (e.g., 'level_0_id_1')
        """
        query = """
        MATCH (c:Community {id: $community_id})
        
        # 1. Capture de toutes les entités de la communauté
        MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
        WITH c, collect(e) AS nodes
        
        # 2. Capture des relations internes (uniquement entre les entités de cette même communauté)
        OPTIONAL MATCH (src)-[r:RELATED]->(tgt)
        WHERE src IN nodes AND tgt IN nodes
        
        # 3. Formatage propre des structures de sortie
        RETURN 
            [n IN nodes | {
                id: n.id, 
                title: n.title, 
                type: n.type, 
                description: coalesce(n.description, "No description available")
            }] AS entities,
            collect(distinct {
                source: src.title, 
                target: tgt.title, 
                description: coalesce(r.description, "No details available")
            }) AS relationships
        """
        try:
            records = await self.client.execute_query(query, {"community_id": community_id})
            if not records:
                return {"entities": [], "relationships": []}
            
            # Neo4j returns a list of maps, we take the first record found
            context_data = records[0]
            
            # Nettoyage si la relation optionnelle a renvoyé un dictionnaire vide
            if context_data["relationships"] == [{"source": None, "target": None, "description": "No details available"}]:
                context_data["relationships"] = []
                
            return context_data
        except Exception as e:
            logger.error(f"❌ Failed to build community context for {community_id}: {e}")
            raise

    async def save_community_report(self, community_id: str, title: str, summary: str, findings: List[Dict[str, Any]]):
        """
        Updates a :Community node with its generated semantic title, executive summary,
        and its structured key findings.

        Args:
            community_id (str): The target community unique ID.
            title (str): The high-level semantic title generated by the LLM.
            summary (str): Executive summary text of the community.
            findings (List[Dict]): A list of dicts containing 'title' and 'explanation'.
        """
        query = """
        MATCH (c:Community {id: $community_id})
        SET c.title = $title,
            c.summary = $summary,
            c.findings_json = $findings_json,
            c.updated_at = timestamp()
        """
        try:
            payload = {
                "community_id": community_id,
                "title": title,
                "summary": summary,
                "findings_json": json.dumps(findings, ensure_ascii=False)
            }
            await self.client.execute_query(query, payload)
            logger.info(f"📝 Successfully updated Community node [{community_id}] with LLM Report.")
        except Exception as e:
            logger.error(f"❌ Failed to save community report for {community_id}: {e}")
            raise