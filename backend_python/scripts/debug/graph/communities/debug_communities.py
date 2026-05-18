from app.infrastructure.neo4j.client import Neo4jClient

import asyncio
import logging

logger = logging.getLogger(__name__)

async def debug_print_community_composition(client: Neo4jClient):
    """
    Query de diagnostic pour inspecter quelles entités 
    composent chaque communauté dans Neo4j.
    """
    query = """
    MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)
    RETURN c.id AS community_id, 
           c.level AS level, 
           collect(e.title) AS entities, 
           count(e) AS size
    ORDER BY size DESC
    """
    try:
        records = await client.execute_query(query)
        print("\n🔎 === DIAGNOSTIC DES COMMUNAUTÉS DETECTÉES ===")
        for r in records:
            print(f"\n📦 [{r['community_id']}] (Level {r['level']}) - Contient {r['size']} entités :")
            print(f"   👉 {', '.join(r['entities'])}")
        print("\n===============================================")
    except Exception as e:
        logger.error(f"Erreur diagnostic : {e}")


async def main():
    client = Neo4jClient()
    await client.connect()
    try:
        await debug_print_community_composition(client)
    except Exception as e:
        logger.error(f"Erreur au lancement: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())