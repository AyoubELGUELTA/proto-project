import logging
from typing import List, Dict
from app.db.neo4j.connection import Neo4jConnection

logger = logging.getLogger(__name__)

async def ingest_graph_data(entities: List[Dict], relations: List[Dict]):
    driver = await Neo4jConnection.get_driver()
    
    async with driver.session() as session:
        # 1. Ingestion des Entités (Utilise MERGE pour éviter les doublons)
        for ent in entities:
            label = ent.get('type', 'Person')
            # Sécurité : On s'assure que le label est une string propre
            query = (
                f"MERGE (n:{label} {{name: $name}}) "
                "SET n.description = $desc, n.aliases = $aliases, n.confidence = $conf"
            )
            await session.run(query, 
                name=ent['name'], 
                desc=ent.get('context_description', ''),
                aliases=ent.get('aliases', []),
                conf=ent.get('confidence', 0.0)
            )
        
        # 2. Ingestion des Relations
        for rel in relations:
            rel_type = rel['relation_type'].replace(" ", "_").upper()
            query = (
                "MATCH (a {name: $source}), (b {name: $target}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $props"
            )
            await session.run(query, 
                source=rel['source_entity'], 
                target=rel['target_entity'], 
                props=rel.get('properties', {})
            )
    logger.info("✅ Ingestion réussie dans Neo4j !")