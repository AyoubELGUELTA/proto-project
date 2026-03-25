


import asyncio
import json
import os
import logging
from app.db.neo4j.queries import ingest_graph_data
from app.db.neo4j.connection import Neo4jConnection

# Configuration du logging pour voir ce qui se passe
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_import():
    # 1. Chemin vers fichier CLEAN
    file_path = "app/test/output/graph_extraction_CLEAN.json"
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Fichier introuvable : {file_path}")
        return

    try:
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entities = data.get("entities", [])
        relations = data.get("relations", [])
        
        logger.info(f"📂 Fichier chargé : {len(entities)} entités et {len(relations)} relations trouvées.")

        # 3. Appel de la fonction d'ingestion (qui contient déjà les MERGE)
        await ingest_graph_data(entities, relations)
        
        logger.info("✨ L'importation est terminée avec succès !")

    except Exception as e:
        logger.error(f"💥 Erreur lors de l'import : {str(e)}")
    finally:
        # 4. On ferme proprement la connexion
        await Neo4jConnection.close()

if __name__ == "__main__":
    # Lancement de la boucle d'événements asynchrone
    asyncio.run(run_import())