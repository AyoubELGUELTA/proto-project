"""
Connexion Neo4j pour Graph RAG
"""

from neo4j import AsyncGraphDatabase
from typing import Optional
from app.core.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  

class Neo4jConnection:
    _driver: Optional[AsyncGraphDatabase] = None
    
    @classmethod
    async def get_driver(cls):
        if cls._driver is None:
            cls._driver = AsyncGraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        return cls._driver
    
    @classmethod
    async def close(cls):
        if cls._driver:
            await cls._driver.close()
            cls._driver = None

async def test_connection():
    driver = await Neo4jConnection.get_driver()
    async with driver.session() as session:
        result = await session.run("RETURN 1 AS test")
        record = await result.single()
        print(f"✅ Neo4j connected: {record['test']}")