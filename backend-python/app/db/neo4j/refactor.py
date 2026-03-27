import logging
from app.db.neo4j.connection import Neo4jConnection
from app.ingestion.graph_prompts import RELATION_CONSOLIDATION_PROMPT
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

async def run_graph_relation_optimization(llm_service: LLMService):
    """
    Orchestrateur du nettoyage : Audit Neo4j -> Mapping LLM -> Mutation Cypher.
    """
    driver = await Neo4jConnection.get_driver()
    
    async with driver.session() as session:
        # 1. AUDIT (Identique)
        logger.info("🔍 Audit des types de relations dans Neo4j...")
        result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        extracted_types = [record["relationshipType"] async for record in result]
        
        if not extracted_types:
            logger.info("ℹ️ Aucune relation trouvée dans la base. Audit stoppé.")
            return

        logger.info(f"📋 Types identifiés dans le graphe : {extracted_types}")

        # 2. LLM : Dispatching System/User pour le Prompt Caching
        logger.info("🤖 Arbitrage en cours par le LLM (Normalisation)...")
        
        # On prépare le payload structuré
        payload = {
            "system": RELATION_CONSOLIDATION_PROMPT,
            "user": f"Voici les relations actuellement présentes dans le graphe Neo4j : {extracted_types}"
        }

        # On passe le dictionnaire au service (qui gère maintenant le dispatch)
        mapping_json = await llm_service.generate_json(payload, show_usage=True)
        
        if not mapping_json or (not mapping_json.get("ALREADY_IN_TAXONOMY") and not mapping_json.get("NOT_IN_TAX_BUT_TO_MERGE")):
            logger.warning("⚠️ Le LLM n'a suggéré aucune fusion ou le format est invalide.")
            return

        # 3. ACTION (Identique)
        logger.info("🏗️ Application du refactoring via APOC...")
        await apply_relation_consolidation(mapping_json)

async def apply_relation_consolidation(mapping_json: dict):
    """
    Applique physiquement les changements de type de relations dans Neo4j.
    """
    driver = await Neo4jConnection.get_driver()
    
    async with driver.session() as session:
        # --- CATÉGORIE 1 : Fusion vers la Taxonomie Officielle ---
        taxonomy = mapping_json.get("ALREADY_IN_TAXONOMY", {})
        for official_type, variants in taxonomy.items():
            for variant in variants:
                # Sécurité : ne pas tenter de fusionner un type vers lui-même
                if variant == official_type:
                    continue
                
                logger.info(f"🔗 Fusion : {variant} ➔ {official_type}")
                # Utilisation de backticks (`) pour gérer les types avec des caractères spéciaux ou espaces
                query = f"""
                MATCH ()-[r:`{variant}`]->()
                CALL apoc.refactor.setType(r, $new_type) YIELD input, output
                RETURN count(output) as count
                """
                await session.run(query, new_type=official_type)
        
        # --- CATÉGORIE 2 : Clustering de synonymes hors taxonomie ---
        merges = mapping_json.get("NOT_IN_TAX_BUT_TO_MERGE", [])
        for group in merges:
            if len(group) < 2:
                continue
            
            # On prend le premier élément comme type de référence (normalisé en UPPER_SNAKE_CASE)
            target_type = group[0].replace(" ", "_").upper()
            variants = group[1:]
            
            for variant in variants:
                if variant == target_type:
                    continue
                    
                logger.info(f"🧩 Clustering : {variant} ➔ {target_type}")
                query = f"""
                MATCH ()-[r:`{variant}`]->()
                CALL apoc.refactor.setType(r, $new_type) YIELD input, output
                RETURN count(output) as count
                """
                await session.run(query, new_type=target_type)

    logger.info("✨ Graphe optimisé et normalisé avec succès.")