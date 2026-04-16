import logging
import pandas as pd
from typing import Dict, Any, List
from app.infrastructure.neo4j.client import Neo4jClient

logger = logging.getLogger(__name__)

class GraphStoreManager:
    """
    Manages the persistence of extracted graph data into Neo4j.
    
    This class transforms Pandas DataFrames (Entities and Relationships)
    into optimized Cypher queries using batch processing (UNWIND).
    """

    def __init__(self, client: Neo4jClient):
        self.client = client

    async def save_graph(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame):
        """
        Main entry point to persist the entire graph batch.
        
        Args:
            entities_df: Resolved entities with 'title' as the primary ID.
            relationships_df: Re-mapped relationships between entities.
        """
        logger.info(f"📤 Pushing graph to Neo4j ({len(entities_df)} nodes, {len(relationships_df)} edges)...")
        
        # 1. Ensure constraints exist before pushing
        await self.client.ensure_constraints()

        # 2. Push Nodes
        if not entities_df.empty:
            await self._upsert_entities(entities_df)

        # 3. Push Relationships
        if not relationships_df.empty:
            await self._upsert_relationships(relationships_df)

        logger.info("✅ Graph successfully synchronized with Neo4j.")

    async def _upsert_entities(self, df: pd.DataFrame):
        """
        Persists entities using a batch UNWIND query.
        
        We use 'id' as the unique identifier (the resolved title or canonical ID).
        """
        # Prepare data: Neo4j prefers a list of dicts
        data = df.to_dict(orient="records")
        
        query = """
        UNWIND $batch AS row
        MERGE (e:Entity {id: row.title})
        SET e.title = row.title,
            e.type = row.type,
            e.description = row.description,
            e.frequency = row.frequency,
            e.updated_at = timestamp()
        """
        await self.client.execute_query(query, parameters={"batch": data})
        logger.debug(f"💎 Upserted {len(data)} Entity nodes.")

    async def _upsert_relationships(self, df: pd.DataFrame):
        """
        Persists relationships between existing entities.
        
        Note: We assume entities already exist thanks to _upsert_entities.
        """
        data = df.to_dict(orient="records")
        
        # We use MERGE to avoid duplicate edges between same nodes
        query = """
        UNWIND $batch AS row
        MATCH (source:Entity {id: row.source})
        MATCH (target:Entity {id: row.target})
        MERGE (source)-[r:RELATED_TO {description: row.description}]->(target)
        SET r.weight = row.weight,
            r.source_id = row.source_id
        """
        await self.client.execute_query(query, parameters={"batch": data})
        logger.debug(f"🔗 Upserted {len(data)} Relationship edges.")