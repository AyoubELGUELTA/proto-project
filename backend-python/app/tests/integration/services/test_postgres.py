import pytest
import os
from app.services.database.postgres_client import PostgresClient
from app.services.database.document_repository import DocumentRepository
from app.services.database.chunk_repository import ChunkRepository
from app.services.database.schema import CREATE_SCHEMA_QUERY
from app.models.domain import TextUnit

@pytest.mark.asyncio
async def test_postgres_full_cycle():
    """
    Test d'intégration qui valide tout le cycle de vie SQL :
    Connexion -> Init Schema -> Create Doc -> Status Update -> Batch Chunks
    """
    # 1. Initialisation du client
    client = PostgresClient()
    await client.connect()
    
    try:
        # 2. Setup Schema
        await client.execute(CREATE_SCHEMA_QUERY)
        
        doc_repo = DocumentRepository(client)
        chunk_repo = ChunkRepository(client)
        
        test_filename = "test_document_v1.pdf"
        
        # 3. Test Get or Create & Status
        doc_id = await doc_repo.get_or_create(test_filename)
        assert doc_id is not None
        print(f"\n✅ Document créé avec l'ID: {doc_id}")
        
        # Passage en mode 'PROCESSING'
        await client.execute("UPDATE documents SET status = 'PROCESSING' WHERE doc_id = $1", doc_id)
        
        # 4. Simulation de TextUnits
        fake_units = [
            TextUnit(
                id="fake_1", 
                text="Contenu du premier chunk", 
                page_numbers=[1],
                metadata={"heading_full": "Introduction"}
            ),
            TextUnit(
                id="fake_2", 
                text="Contenu du second chunk", 
                page_numbers=[1, 2],
                metadata={"heading_full": "Méthodologie", "image_urls": ["http://minio/img1.jpg"]}
            )
        ]
        
        # 5. Test de l'insertion en Batch (copy_records_to_table)
        await chunk_repo.store_text_units(doc_id, fake_units)
        
        # 6. Vérification finale
        count = await client.fetchval("SELECT COUNT(*) FROM chunks WHERE doc_id = $1", doc_id)
        assert count == 2
        
        # Update final
        await client.execute("UPDATE documents SET status = 'COMPLETED' WHERE doc_id = $1", doc_id)
        print(f"✅ Cycle Postgres terminé avec succès : {count} chunks insérés.")

    finally:
        # On nettoie pour ne pas polluer les futurs tests (Optionnel)
        # await client.execute("DELETE FROM documents WHERE filename = $1", test_filename)
        await client.disconnect()