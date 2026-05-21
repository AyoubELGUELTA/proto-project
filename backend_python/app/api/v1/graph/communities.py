import logging
from fastapi import APIRouter, BackgroundTasks
from app.infrastructure.neo4j.client import Neo4jClient
from app.services.graph.community_service import CommunityService
from app.indexing.operations.communities.context_builder import HierarchicalContextBuilder
from app.indexing.operations.communities.report_builder import CommunityReportBuilder
from app.services.llm.factory import LLMFactory
from app.indexing.workflows.community_reporting_pipeline import run_community_reporting_workflow 

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/communities", 
    tags=["Graph Communities"]
)

async def _reporting_background_worker(neo4j_client: Neo4jClient):
    """
    Worker isolé qui s'exécute en tâche de fond. 
    Gère le cycle de vie du workflow, le tracking des coûts et la fermeture des connexions.
    """
    try:
        # 1. Initialisation de la brigade de services
        community_service = CommunityService(neo4j_client)
        context_builder = HierarchicalContextBuilder(community_service=community_service)
        
        # Récupération du service LLM spécifiquement calibré pour les rapports de communautés 🎯
        community_reporter_llm = LLMFactory.get_community_reporting_service() 
        report_builder = CommunityReportBuilder(llm=community_reporter_llm)
        
        # 2. Exécution du workflow lourd
        await run_community_reporting_workflow(
            community_service=community_service,
            context_builder=context_builder,
            report_builder=report_builder
        )
        
        # 3. Récupération et affichage du rapport financier du Tracker
        tracker = LLMFactory.get_tracker()
        final_report = tracker.get_report()
        
        logger.info("==================================================================")
        logger.info("💰 COMMUNITY REPORTING PIPELINE - COST & USAGE REPORT 💰")
        logger.info(f"🔹 Total Tokens Consumed: {tracker.usage.total_tokens:,}")
        logger.info(f"🔹 Total Pipeline Cost:  ${tracker.usage.total_cost:.5f} USD")
        logger.info(f"🔹 Detailed Usage Breakdown:\n{final_report}")
        logger.info("==================================================================")

    except Exception as e:
        logger.critical(f"💥 Background reporting workflow failed: {str(e)}", exc_info=True)
    finally:
        # Sécurité cruciale : On ferme la connexion une fois que la tâche de fond est FINIE
        await neo4j_client.close()
        logger.info("🔌 Neo4j client connection closed gracefully from background task.")


@router.post("/refresh-reports")
async def trigger_community_refresh(background_tasks: BackgroundTasks):
    """
    Manually triggers the elite hierarchical community reporting pipeline.
    Runs asynchronously as a background task to prevent HTTP timeouts.
    """
    logger.info("⚡ API Request received to refresh community reports.")
    
    # On ouvre la connexion ici pour s'assurer qu'elle est prête avant de lancer le thread de fond
    neo4j_client = Neo4jClient()
    await neo4j_client.connect()
    
    # On délègue l'exécution globale au worker en tâche de fond
    background_tasks.add_task(_reporting_background_worker, neo4j_client=neo4j_client)
    
    return {
        "status": "processing",
        "message": "Hierarchical community reporting pipeline successfully triggered in the background. Check server logs for cost report upon completion."
    }