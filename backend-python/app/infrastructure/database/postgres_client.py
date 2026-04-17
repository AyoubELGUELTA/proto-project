import asyncpg
import logging
from typing import Optional, List, Any
from app.core.settings import settings  

logger = logging.getLogger(__name__)

class PostgresClient:
    """
    Asynchronous PostgreSQL client utilizing connection pooling for high-performance RAG operations.
    
    This client manages a pool of connections to handle concurrent data ingestion 
    and retrieval, ensuring low latency and efficient resource management.
    """

    def __init__(self):
        self.host = settings.db_host
        self.port = settings.db_port
        self.database = settings.db_name
        self.user = settings.db_user
        self.password = settings.db_password
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """
        Initializes the connection pool.
        
        Configurations:
            min_size=5: Keeps 5 connections ready at all times.
            max_size=20: Allows scaling up to 20 concurrent queries during heavy ingestion.
        """
        if not self._pool:
            try:
                logger.info(f"📡 Connecting to DB: user={self.user} at {self.host}:{self.port}/{self.database}")
                
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    min_size=5,
                    max_size=20
                )
                logger.info("✅ Database connection pool established.")
            except Exception as e:
                logger.critical(f"❌ Failed to create Postgres pool: {e}")
                raise e

    async def disconnect(self):
        """Closes the connection pool and releases all resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("🔌 Database pool closed.")

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Executes a query and returns all resulting rows.

        Args:
            query (str): The SQL statement.
            *args: Parameters for the query.
        """
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """
        Executes a query and returns the value of the first column of the first row.
        Useful for retrieving IDs (RETURNING doc_id).
        """
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args) -> str:
        """
        Executes a command (INSERT, UPDATE, DELETE) and returns the status string.
        """
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)