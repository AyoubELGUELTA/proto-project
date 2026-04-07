from fastapi import UploadFile, BackgroundTasks


# Ingestion + DB related
from app.services.database.postgres_client import PostgresClient
from app.services.database.document_repository import DocumentRepository
from app.services.database.chunk_repository import ChunkRepository
from app.services.database.ingestion_context import IngestionContext
from app.services.storage.file_service import FileService
from app.services.llm.service import LLMService

from app.indexing.operations.text.identity_service import IdentityService
from app.indexing.workflows.create_text_units import workflow_create_text_units

# Graph related
from app.indexing.operations.graph.extract_graph import GraphExtractor
from app.indexing.operations.entity_resolution.core_resolver import CoreResolver
from app.indexing.operations.entity_resolution.llm_resolver import LLMResolver
from app.services.llm.parser import LLMParser
from app.services.graph.graph_service import GraphService
from app.indexing.workflows.extract_graph import extract_graph as workflow_extract_graph

# Model
from app.models.domain import TextUnit

#Others

import logging

logger = logging.getLogger(__name__)


async def ingest_single_file(file: UploadFile):
    # INIT SERVICES
    db = PostgresClient()
    await db.connect()
    
    llm_service = LLMService() 
    
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    file_service = FileService()
    identity_service = IdentityService(llm_service)

    graph_service = GraphService(
        extractor=GraphExtractor(llm_service),
        summarizer=llm_service.summarize_entities, # Ta méthode de résumé
        parser=LLMParser(),
        core_resolver=CoreResolver(path="data/encyclopedia.json"),
        llm_resolver=LLMResolver(llm_service)
    )

    # PRÉPARATION
    doc_id = await doc_repo.get_or_create(file.filename)

    async with IngestionContext(doc_repo, doc_id):
        
        # A. Sauvegarde physique
        local_path = await file_service.save_uploaded_file(file, doc_id)

        # B. Workflow Indexing (Docling) -> Récupère les TextUnits
        final_units = await workflow_create_text_units(local_path)

        # C. Persistance des Chunks "Content"
        await chunk_repo.store_text_units(doc_id, final_units, chunk_type="CONTENT")

        # D. IA Enrichment: Identity Card
        # On génère la fiche à partir de l'échantillonnage intelligent
        identity_data = await identity_service.generate_identity(final_units)
        
        # Sauvegarde 1 : Dans la table 'documents' (Metadata JSON)
        await doc_repo.update_metadata(doc_id, identity_data)
        
        # Sauvegarde 2 : Dans la table 'chunks' comme un type 'IDENTITY'
        # On crée une TextUnit fictive pour wrapper le résumé pour le RAG
        identity_unit = TextUnit(
            id=f"id_{doc_id}",
            text=identity_data.get("executive_summary", ""),
            metadata=identity_data
        )
        await chunk_repo.store_text_units(doc_id, [identity_unit], chunk_type="IDENTITY")

        # --- E. GRAPH EXTRACTION (La nouvelle étape) ---
        # On passe le contexte (Executive Summary) pour guider l'extraction
        domain_context = identity_data.get("executive_summary", "A general historical document Islam-related, or Academic-related.")
        
        # Lancement du workflow qu'on a construit ensemble
        entities_df, relationships_df = await workflow_extract_graph(
            text_units=final_units,
            graph_service=graph_service,
            domain_context=domain_context
        )

        # --- F. PERSISTENCE DU GRAPHE ---
        # Ici, on appellera le futur GraphStore pour save en DB
        # await graph_store.save_graph(doc_id, entities_df, relationships_df)
        
        logger.info(f"Graph created: {len(entities_df)} entities, {len(relationships_df)} relations.")

    # 4. FERMETURE
    await db.disconnect()
    return {
        "status": "success", 
        "doc_id": doc_id, 
        "stats": {
            "entities": len(entities_df),
            "relations": len(relationships_df)
        }
    }