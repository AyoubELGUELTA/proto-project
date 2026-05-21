import logging
from fastapi import UploadFile, APIRouter
from typing import Dict, Any

# Services & Repositories
from app.infrastructure.database.postgres_client import PostgresClient
from app.infrastructure.neo4j.client import Neo4jClient
from app.services.database.document_repository import DocumentRepository
from app.services.database.chunk_repository import ChunkRepository
from app.services.database.encyclopedia_repository import EncyclopediaRepository
from app.services.database.ingestion_context import IngestionContext
from app.services.storage.file_service import FileService
from app.services.llm.factory import LLMFactory
from app.services.llm.parser import LLMParser
from app.services.graph.graph_service import GraphService
from app.services.graph.community_service import CommunityService

# Resolution Engine & Operations
from app.indexing.operations.text.identity_service import IdentityService
from app.indexing.workflows.create_text_units import workflow_create_text_units
from app.indexing.operations.graph.graph_extractor import EntityAndRelationExtractor
from app.indexing.operations.graph.summarize_manager import SummarizeManager 
from app.indexing.operations.graph.store_manager import GraphStoreManager
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine

from app.core.data_model.text_units import TextUnit

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Ingestion"])
@router.post("/ingest")
async def ingest_single_file(file: UploadFile) -> Dict[str, Any]:
    """
    Orchestrates the complete ingestion pipeline for a single PDF document.
    Utilizes task-specific decoupled LLM services via LLMFactory.
    """
    
    logger.info(f"📥 Starting ingestion pipeline for: {file.filename}")
    
    # 1. Initialize Infrastructure
    db = PostgresClient()
    await db.connect()

    neo4j_client = Neo4jClient()
    await neo4j_client.connect()

    file_service = FileService()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    encyclopedia_repo = EncyclopediaRepository(db)
    store_manager = GraphStoreManager(neo4j_client)

    # 2. ASSEMBLE RESOLUTION ENGINE (Multiplication des attributs ici 🎯)
    core_res = CoreResolver(encyclopedia=EncyclopediaManager(encyclopedia_repo))
    
    llm_res = LLMResolver(
        entity_resolution_service=LLMFactory.get_entity_resolution_service(),
        anchoring_resolution_service=LLMFactory.get_anchoring_resolution_service(),
        consultant_resolution_service=LLMFactory.get_consultant_resolution_service()
    )
    res_engine = EntityResolutionEngine(core_resolver=core_res, llm_resolver=llm_res)

    # 3. ASSEMBLE GRAPH SERVICES (Injection des services granulaires 🎯)
    community_service = CommunityService(neo4j_client)

    graph_service = GraphService(
        extractor=EntityAndRelationExtractor(LLMFactory.get_graph_extraction_service()),
        summarizer=SummarizeManager(LLMFactory.get_element_summarization_service()), 
        parser=LLMParser(),
        resolution_engine=res_engine,
        store_manager=store_manager,
        community_service=community_service
    )

    identity_service = IdentityService(LLMFactory.get_document_identity_service())

    # 4. DOCUMENT PREPARATION
    doc_id = await doc_repo.get_or_create(file.filename)
    logger.info(f"📄 Document registered with ID: {doc_id}")

    try:
        async with IngestionContext(doc_repo, doc_id):
            
            # A. Physical Ingestion & Parsing (Docling + Spatial)
            local_path = await file_service.save_uploaded_file(file, doc_id)
            final_units = await workflow_create_text_units(local_path)

            # B. Persist Chunks to SQL
            await chunk_repo.store_text_units(doc_id, final_units, chunk_type="CONTENT")

            # C. Identity Card Generation
            logger.info("🪪 Generating Document Identity Card...")
            identity_data = await identity_service.generate_identity(final_units)
            await doc_repo.update_metadata(doc_id, identity_data)
            
            # Create a virtual unit for the Identity Card (useful for global RAG context)
            identity_unit = TextUnit(
                id=f"id_{doc_id}",
                text=identity_data.get("executive_summary", ""),
                metadata=identity_data
            )
            await chunk_repo.store_text_units(doc_id, [identity_unit], chunk_type="IDENTITY")

            # D. FULL GRAPH LIFECYCLE (Extraction, Resolution, Summarization & Community Clustering)
            domain_context = identity_data.get("executive_summary", "A general historical document.")
            
            logger.info("🕸️ Running Graph Extraction pipeline...")
            entities_df, relationships_df = await graph_service.run_pipeline(
                text_units=final_units,
                domain_context=domain_context
            )
            
            logger.info(f"✅ Graph success: {len(entities_df)} entities, {len(relationships_df)} relations.")

        # 5. FINAL REPORTING
        tracker = LLMFactory.get_tracker()
        final_report = tracker.get_report()
        
        await db.disconnect()
        await neo4j_client.close()
        
        return {
            "status": "success", 
            "doc_id": doc_id, 
            "graph": {
                "entities": entities_df.to_dict(orient="records"),
                "relationships": relationships_df.to_dict(orient="records")
            },
            "stats": {
                "entities_count": len(entities_df),
                "relations_count": len(relationships_df),
                "total_tokens": tracker.usage.total_tokens,
                "total_cost_usd": tracker.usage.total_cost,
                "detailed_report": final_report
            }
        }

    except Exception as e:
        logger.critical(f"💥 Pipeline failed for doc {doc_id}: {str(e)}", exc_info=True)
        await db.disconnect()
        return {"status": "error", "message": str(e), "doc_id": doc_id}