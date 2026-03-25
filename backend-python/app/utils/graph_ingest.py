import json
import logging
from app.db.neo4j.connection import Neo4jConnection

logger = logging.getLogger(__name__)

class GraphIngestor:
    @staticmethod
    async def run_ingestion(json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        driver = await Neo4jConnection.get_driver()
        
        async with driver.session() as session:
            # 1. Création des contraintes (pour la performance et l'unicité)
            # Note: On le fait de manière générique sur les types rencontrés
            entity_types = set(e['type'] for e in data['entities'])
            for etype in entity_types:
                try:
                    await session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{etype}) REQUIRE n.name IS UNIQUE")
                except Exception as e:
                    logger.warning(f"Could not create constraint for {etype}: {e}")

            # 2. Ingestion des Entités
            logger.info(f"🚀 Ingesting {len(data['entities'])} entities...")
            for ent in data['entities']:
                # On utilise une requête dynamique pour le Label
                query = (
                    f"MERGE (n:{ent['type']} {{name: $name}}) "
                    "SET n.description = $desc, "
                    "    n.aliases = $aliases, "
                    "    n.confidence = $conf "
                    "RETURN n"
                )
                await session.run(query, 
                    name=ent['name'], 
                    desc=ent.get('context_description', ''),
                    aliases=ent.get('aliases', []),
                    conf=ent.get('confidence', 1.0)
                )

            # 3. Ingestion des Relations
            logger.info(f"🔗 Ingesting {len(data['relations'])} relations...")
            for rel in data['relations']:
                # On cherche les nœuds sans connaître leurs labels à l'avance (MATCH (n {name:...}))
                # Puis on crée la relation typée
                rel_type = rel['relation_type'].replace(" ", "_").upper()
                query = (
                    "MATCH (source {name: $source_name}) "
                    "MATCH (target {name: $target_name}) "
                    f"MERGE (source)-[r:{rel_type}]->(target) "
                    "SET r += $props "
                    "RETURN r"
                )
                await session.run(query, 
                    source_name=rel['source_entity'],
                    target_name=rel['target_entity'],
                    props=rel.get('properties', {})
                )
            
            logger.info("✅ Ingestion Neo4j terminée avec succès.")