from fastapi import UploadFile, BackgroundTasks

from app.services.database.postgres_client import PostgresClient
from app.services.database.document_repository import DocumentRepository
from app.services.database.chunk_repository import ChunkRepository
from app.services.database.ingestion_context import IngestionContext
from app.services.storage.file_service import FileService
from app.services.llm.service import LLMService

from app.indexing.operations.text.identity_service import IdentityService
from app.indexing.workflows.create_text_units import workflow_create_text_units

from app.models.domain import TextUnit

async def ingest_single_file(file: UploadFile):
    # INIT SERVICES
    db = PostgresClient()
    await db.connect()
    
    # On imagine un LLMService déjà configuré (OpenAI/Mistral)
    llm_service = LLMService() 
    
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    file_service = FileService()
    identity_service = IdentityService(llm_service)

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

        # E. Prochaine étape : Graph Extraction (On garde doc_id et identity_data)
        # ... 

    # 4. FERMETURE
    await db.disconnect()
    return {"status": "success", "doc_id": doc_id, "identity": identity_data}