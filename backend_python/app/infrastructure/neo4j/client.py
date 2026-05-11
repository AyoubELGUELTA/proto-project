import logging
from typing import Any, Dict, List, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from app.core.settings import settings 

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Core infrastructure client for Neo4j operations.
    
    This class manages the lifecycle of the AsyncDriver and provides 
    low-level methods to interact with the graph database while 
    ensuring proper resource cleanup and error handling.
    """

    def __init__(self):
        self._driver: Optional[AsyncDriver] = None
        self._uri = settings.neo4j_uri
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password

    async def connect(self):
        """Initializes the Neo4j AsyncDriver and verifies connectivity."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._password)
            )
            await self._driver.verify_connectivity()
            logger.info("✅ Successfully connected to Neo4j instance.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise

    async def close(self):
        """Closes the driver connection pool."""
        if self._driver:
            await self._driver.close()
            logger.info("🔌 Neo4j connection closed.")

    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: str = "neo4j"
    ) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query within a single asynchronous session.
        
        Args:
            query: The Cypher query string.
            parameters: Dictionary of query parameters.
            database: Target database name.
            
        Returns:
            A list of records as dictionaries.
        """
        if not self._driver:
            raise RuntimeError("Neo4j Driver is not initialized. Call connect() first.")

        try:
            records, _, _ = await self._driver.execute_query(
                query, 
                parameters_=parameters,
                database_=database
            )
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"❌ Cypher Query Error: {e}\nQuery: {query}")
            raise

    async def ensure_constraints(self):
        """
        Sets up database constraints and indexes.
        
        Guarantees that each Entity is unique based on its 'id' (canonical name).
        """
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX entity_title_index IF NOT EXISTS FOR (e:Entity) ON (e.title)"
        ]
        for cypher in constraints:
            await self.execute_query(cypher)
            logger.debug(f"⚙️ Applied Neo4j constraint/index: {cypher.split('FOR')[0].strip()}")