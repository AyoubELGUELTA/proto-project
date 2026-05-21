import logging
from app.services.graph.community_service import CommunityService
from app.indexing.operations.communities.context_builder import HierarchicalContextBuilder
from app.indexing.operations.communities.report_builder import CommunityReportBuilder
from app.indexing.operations.communities.utils import resolve_context_references
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
    logger.info("⚡ [WORKFLOW START] Entering run_community_reporting_workflow background task.")
    logger.info(f"⚙️ Configuration | Drift Threshold: {drift_threshold}")

    try:
        # 1. Récupération du manifest
        logger.info("📡 Step 1: Fetching analysis manifest from CommunityService...")
        manifest = await community_service.get_communities_analysis_manifest(drift_threshold=drift_threshold)
        
        if manifest is None:
            logger.error("❌ Step 1 Critical: Manifest returned 'None'. Check community_service connection.")
            return
            
        if not manifest:
            logger.info("✅ Step 1 Complete: All communities are perfectly synced. Nothing to process.")
            return

        logger.info(f"📊 Step 1 Complete: Found {len(manifest)} raw community targets needing updates.")

        # 2. Tri multi-critères
        logger.info("🔀 Step 2: Sorting targets bottom-up (Hierarchical level DESC, drift score DESC)...")
        manifest.sort(key=lambda x: (-x.get("level", 0), -x["score"]))
        logger.info(f"🎯 Step 2 Complete: Process order established for {len(manifest)} target communities.")

        # 3. Boucle de traitement
        processed_count = 0
        for index, task in enumerate(manifest, start=1):
            comm_id = task["id"]
            comm_level = task.get("level", 0)
            
            logger.info(f"🔄 [LOOP {index}/{len(manifest)}] Starting pipeline for community: {comm_id} (Level {comm_level})")
            
            try:
                # A. Contexte (On unpack le tuple : texte + dictionnaire de mapping)
                logger.info(f"   ⏳ [{comm_id}] A. Building optimized hybrid context...")
                optimized_context, id_mapping = await context_builder.build_optimized_context(community_id=comm_id)
                logger.info(f"   ✅ [{comm_id}] A. Context ready. Char length: {len(optimized_context)}")

                # B. Génération LLM
                logger.info(f"   ⏳ [{comm_id}] B. Dispatching context to LLM for report generation...")
                report_schema = await report_builder.generate_report(
                    community_id=comm_id, 
                    optimized_context=optimized_context
                )
                logger.info(f"   ✅ [{comm_id}] B. LLM structured report received. Title: '{report_schema.title}'")

                # C. Mapping et Résolution des Données (La Magie opère ici ✨)
                logger.info(f"   ⏳ [{comm_id}] C. Dumping models and resolving local IDs to global UUIDs...")
                
                findings_dicts = []
                for finding in report_schema.findings:
                    finding_dict = finding.model_dump()
                    
                    # On intercepte l'explication et on remplace les chiffres par les vrais IDs de la base !
                    raw_explanation = finding_dict.get("explanation", "")
                    resolved_explanation = resolve_context_references(raw_explanation, id_mapping)
                    
                    finding_dict["explanation"] = resolved_explanation
                    findings_dicts.append(finding_dict)
                
                state_metadata = {
                    "hash": task["target_hash"],
                    "entity_count": task["entity_count"],
                    "relationship_count": task["relationship_count"],
                    "semantic_mass": task["semantic_mass"],
                    "rating": report_schema.rating,
                    "rating_explanation": report_schema.rating_explanation
                }

                # D. Persistance Neo4j
                logger.info(f"   ⏳ [{comm_id}] D. Persisting report with resolved global IDs in Neo4j...")
                await community_service.save_community_report(
                    community_id=comm_id,
                    title=report_schema.title,
                    summary=report_schema.summary,
                    findings=findings_dicts,
                    state_metadata=state_metadata
                )
                logger.info(f"   💾 [{comm_id}] D. Report locked and state synced successfully.")
                processed_count += 1

            except Exception as e:
                logger.error(f"💥 Error processing community {comm_id}: {e}", exc_info=True)
                raise e

        logger.info(f"🏁 [WORKFLOW END] Pipeline finished. Successfully updated {processed_count}/{len(manifest)} communities.")

    except Exception as global_env_error:
        logger.critical(f"🚨 [GLOBAL WORKFLOW CRASH] Task failed before loop execution: {global_env_error}", exc_info=True)

    logger.info("🏁 Elite Hierarchical Community Workflow completed with success.")