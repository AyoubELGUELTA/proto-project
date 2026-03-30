import logging
from typing import List, Dict
from app.db.neo4j.connection import Neo4jConnection

logger = logging.getLogger(__name__)

async def ingest_graph_data(entities: List[Dict], relations: List[Dict]):
    driver = await Neo4jConnection.get_driver()
    async with driver.session() as session:
        
        # 1. Traitement des Entités

        entities_batch = []
        for ent in entities:
            raw_type = ent.get('type', 'Location')
            # Récupération dynamique : ex ["Prophet", "Human", "Man"]
            labels = EntityType.get_all_labels(raw_type)
            
            entities_batch.append({
                "id": ent['normalized_name'],
                "name": ent['name'],
                "labels": labels,
                "desc": ent.get('context_description', ''),
                "aliases": ent.get('aliases', []),
                "conf": ent.get('confidence', 0.0)
            })

        # Query optimisée : UNWIND sur toute la liste d'un coup
        entity_query = """
        UNWIND $batch AS data
        MERGE (n:Entity {id: data.id})
        SET n.display_name = data.name,
            n.description = data.desc,
            n.aliases = data.aliases,
            n.last_confidence = data.conf
            
        // Ajout dynamique des labels (Héritage)
        // apoc.create.addLabels n'écrase pas, il ajoute.
        WITH n, data
        CALL apoc.create.addLabels(n, data.labels) YIELD node
        RETURN count(node)
        """
        await session.run(entity_query, batch=entities_batch)

        # 2. Traitement des Relations
        rel_batch = []
        for rel in relations:
            rel_batch.append({
                "s": rel.get('source_id'), 
                "t": rel.get('target_id'),
                "type": rel.get('type', 'RELATED_TO').replace(" ", "_").upper(),
                "props": rel.get('properties', {}),
                "evidence": rel.get('evidence', '')
            })

        relationship_query = """
        UNWIND $rel_batch AS r
        MATCH (a:Entity {id: r.s})
        MATCH (b:Entity {id: r.t})
        
        CALL apoc.merge.relationship(a, r.type, {}, r.props, b) YIELD rel
        
        SET rel.occurrence_count = coalesce(rel.occurrence_count, 0) + 1,
            rel.evidences = apoc.coll.toSet(coalesce(rel.evidences, []) + [r.evidence])
        """
        await session.run(relationship_query, rel_batch=rel_batch)