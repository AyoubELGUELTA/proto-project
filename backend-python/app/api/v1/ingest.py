from fastapi import UploadFile
import logging

# Services
from app.services.database.postgres_client import PostgresClient
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
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.indexing.operations.entity_resolution.resolution_engine import EntityResolutionEngine

from app.models.domain import TextUnit

logger = logging.getLogger(__name__)

async def ingest_single_file(file: UploadFile):
    # 1. INIT BASE SERVICES
    db = PostgresClient()
    await db.connect()
    
    # On récupère les deux cerveaux (Light pour l'extraction massive, Heavy pour la synthèse/identité)
    # Ils partagent le même tracker via LLMFactory
    llm_light = LLMFactory.get_light_extractor() 
    llm_heavy = LLMFactory.get_heavy_extractor() 

    file_service = FileService()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    parser = LLMParser()
    
    # 2. ASSEMBLAGE DU MOTEUR DE RÉSOLUTION (Le cerveau logique)
    core_res = CoreResolver(encyclopedia=EncyclopediaManager())
    llm_res = LLMResolver(llm_light)
    res_engine = EntityResolutionEngine(core_resolver=core_res, llm_resolver=llm_res)

    # 3. ASSEMBLAGE DU GRAPH SERVICE
    # On lui donne tout ce qu'il faut pour travailler en autonomie
    graph_service = GraphService(
        extractor=EntityAndRelationExtractor(llm_light),
        summarizer=SummarizeManager(llm_light), 
        parser=parser,
        resolution_engine=res_engine
    )

    identity_service = IdentityService(llm_heavy) # L'identité utilise le modèle puissant

    # 4. PRÉPARATION DU DOCUMENT
    doc_id = await doc_repo.get_or_create(file.filename)

    async with IngestionContext(doc_repo, doc_id):
        
        # A. Ingestion physique & Parsing Document (Docling)
        local_path = await file_service.save_uploaded_file(file, doc_id)
        final_units = await workflow_create_text_units(local_path)

        # B. Persistance Chunks
        await chunk_repo.store_text_units(doc_id, final_units, chunk_type="CONTENT")

        # C. Identity Card (Context pour le graphe)
        identity_data = await identity_service.generate_identity(final_units)
        await doc_repo.update_metadata(doc_id, identity_data)
        
        # Wrap pour le RAG
        identity_unit = TextUnit(
            id=f"id_{doc_id}",
            text=identity_data.get("executive_summary", ""),
            metadata=identity_data
        )
        await chunk_repo.store_text_units(doc_id, [identity_unit], chunk_type="IDENTITY")

        # --- D. EXTRACTION DU GRAPHE ---
        domain_context = identity_data.get("executive_summary", "A general historical document.")

        entities_df, relationships_df = await graph_service.run_pipeline(
            text_units=final_units,
            domain_context=domain_context
        )
        
        logger.info(f"Graph success: {len(entities_df)} entities, {len(relationships_df)} relations.")

    # 5. CONSOMMATION & RÉPONSE
    tracker = LLMFactory.get_tracker()
    
    # On récupère le rapport final qui contient TOUT (Identity + Extraction + Résolution + Summary)
    final_report = tracker.get_report()
    
    await db.disconnect()
    
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