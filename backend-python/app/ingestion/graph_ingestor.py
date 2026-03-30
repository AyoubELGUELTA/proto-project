import json
import aiofiles
from app.db.neo4j.queries import ingest_graph_data

async def run_neo4j_ingestion_from_file(json_path: str):
    """
    Lit le fichier JSON et délègue l'ingestion aux queries Neo4j.
    """
    async with aiofiles.open(json_path, mode='r', encoding='utf-8') as f:
        content = await f.read()
        data = json.loads(content)
        
    entities = data.get("entities", [])
    relations = data.get("relations", [])
    
    if not entities:
        return {"status": "skipped", "message": "No entities found"}

    await ingest_graph_data(entities, relations)
    return {"status": "success", "entities_count": len(entities)}