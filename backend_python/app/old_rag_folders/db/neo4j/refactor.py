import logging
from app.db.neo4j.connection import Neo4jConnection
from app.ingestion.graph_prompts import RELATION_CONSOLIDATION_PROMPT
from app.ingestion.graph_schemas import RelationType
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

async def run_graph_relation_optimization(llm_service: LLMService):
    driver = await Neo4jConnection.get_driver()
    
    async with driver.session() as session:
        # 1. Récupération des types réels en base
        result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        db_types = {record["relationshipType"] async for record in result} # On utilise un set {}
        
        # 2. Récupération de ta taxonomie officielle
        official_set = {r.value for r in RelationType}

        # 3. LE MINUS : On ne garde que ce qui n'est PAS officiel
        # types_to_audit = Ce qui est en DB mais PAS dans l'officiel
        types_to_audit = db_types - official_set

        if not types_to_audit:
            logger.info("✨ Graphe déjà 100% conforme à la taxonomie. Audit stoppé.")
            return

        # 4. Préparation du prompt uniquement avec les suspects
        formatted_extracted = "\n".join([f"- {t}" for t in types_to_audit])
        formatted_taxonomy = ", ".join(list(official_set))
        
        payload = {
            "system": RELATION_CONSOLIDATION_PROMPT.format(
                official_taxonomy=formatted_taxonomy,
                extracted_relations=formatted_extracted 
            ),
            "user": "Analyse ces types inconnus et propose un mapping vers la taxonomie ou des fusions entre eux."
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