import logging
from backend_python.app.services.graph.community_service import CommunityService
from backend_python.app.indexing.operations.communities.context_builder import HierarchicalContextBuilder
from app.indexing.operations.communities.report_builder import CommunityReportBuilder

logger = logging.getLogger(__name__)

async def run_community_reporting_workflow(
    community_service: CommunityService,
    context_builder: HierarchicalContextBuilder,
    report_builder: CommunityReportBuilder,
    drift_threshold: float = 0.15
):
    """
    Elite Level-by-Level Hierarchical Community Reporting Pipeline.
    Processes graph from Leaf Levels to Root Levels to allow semantic report substitution.
    """
    logger.info("🚀 Triggering Elite Hierarchical Community Reporting Pipeline...")

    # 1. Récupération du manifest trié par urgence
    manifest = await community_service.get_communities_analysis_manifest(drift_threshold=drift_threshold)
    if not manifest:
        logger.info("✅ All communities are perfectly synced. Nothing to process.")
        return

    # 2. Tri multi-critères :
    # A. level décroissant (x["level"] inversé avec le signe moins pour trier du plus grand au plus petit)
    # B. score décroissant (les plus gros changements d'abord au sein du même niveau)
    manifest.sort(key=lambda x: (-x.get("level", 0), -x["score"]))

    logger.info(f"🎯 Processing {len(manifest)} targets ordered bottom-up for optimal substitution.")

    for task in manifest:
        comm_id = task["id"]
        comm_level = task.get("level", 0)
        
        try:
            logger.info(f"📋 Building context for {comm_id} (Level {comm_level}) | Drift: {task['score']:.2%}")

            # A. Construction intelligente du contexte (Calibre Microsoft avec substitution)
            optimized_context = await context_builder.build_optimized_context(
                community_id=comm_id, 
            )

            # B. Génération du rapport structuré
            report_schema = await report_builder.generate_report(
                community_id=comm_id, 
                optimized_context=optimized_context
            )

            # C. Préparation des données pour Neo4j (On inclut la notation de Microsoft)
            findings_dicts = [f.model_dump() for f in report_schema.findings]
            
            state_metadata = {
                "hash": task["target_hash"],
                "entity_count": task["entity_count"],
                "relationship_count": task["relationship_count"],
                "semantic_mass": task["semantic_mass"], #TODO check if semantic_mass is the exact name
                "rating": report_schema.rating,
                "rating_explanation": report_schema.rating_explanation
            }

            # D. Sauvegarde en base de données
            await community_service.save_community_report(
                community_id=comm_id,
                title=report_schema.title,
                summary=report_schema.summary,
                findings=findings_dicts,
                state_metadata=state_metadata
            )
            logger.info(f"💾 Report successfully locked in DB for community {comm_id}.")

        except Exception as e:
            logger.error(f"❌ Failed processing for community {comm_id}: {e}", exc_info=True)
            continue

    logger.info("🏁 Elite Hierarchical Community Workflow completed with success.")