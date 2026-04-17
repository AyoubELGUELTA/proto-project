import logging
from fastapi import UploadFile
from typing import Dict, Any

# Services & Repositories
from app.infrastructure.database.postgres_client import PostgresClient
from app.infrastructure.neo4j.client import Neo4jClient
from app.services.database.document_repository import DocumentRepository
from app.services.database.chunk_repository import ChunkRepository
from app.services.database.ingestion_context import IngestionContext
from app.services.storage.file_service import FileService
from app.services.llm.factory import LLMFactory
from app.services.llm.parser import LLMParser
from app.services.graph.graph_service import GraphService

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

from app.models.domain import TextUnit

logger = logging.getLogger(__name__)

async def ingest_single_file(file: UploadFile) -> Dict[str, Any]:
    """
    Orchestrates the complete ingestion pipeline for a single PDF document.
    
    Pipeline Steps:
    1. Infrastructure Setup: Connects to DB and initializes LLM services.
    2. Document Staging: Saves file locally and creates record in PostgreSQL.
    3. Text Unitization: Layout-aware chunking and spatial enrichment.
    4. Identity Generation: Creates a high-level "Identity Card" for global context.
    5. Graph Extraction: Distributed extraction of entities and relationships.
    6. Entity Resolution: Merges duplicates using core logic and LLM verification.
    7. Persistence: Stores chunks and metadata (Graph storage usually follows).

    Args:
        file (UploadFile): The raw PDF file from the API request.

    Returns:
        Dict[str, Any]: A summary of the ingestion results, including graph stats and cost report.
    """
    logger.info(f"📥 Starting ingestion pipeline for: {file.filename}")
    
    # 1. Initialize Infrastructure
    db = PostgresClient()
    await db.connect()

    neo4j_client = Neo4jClient()
    await neo4j_client.connect()

    
    # Semantic LLM selection: Light for volume, Heavy for reasoning
    
    llm_light = LLMFactory.get_light_extractor() 
    llm_heavy = LLMFactory.get_heavy_extractor() 

    file_service = FileService()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    parser = LLMParser()
    
    store_manager = GraphStoreManager(neo4j_client)

    # ASSEMBLE RESOLUTION ENGINE
    core_res = CoreResolver(encyclopedia=EncyclopediaManager())
    llm_res = LLMResolver(llm_light)
    res_engine = EntityResolutionEngine(core_resolver=core_res, llm_resolver=llm_res)


    # ASSEMBLE GRAPH SERVICE
    graph_service = GraphService(
        extractor=EntityAndRelationExtractor(llm_light),
        summarizer=SummarizeManager(llm_light), 
        parser=parser,
        resolution_engine=res_engine,
        store_manager=store_manager

    )

    identity_service = IdentityService(llm_heavy)

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

            # D. GRAPH EXTRACTION & RESOLUTION
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