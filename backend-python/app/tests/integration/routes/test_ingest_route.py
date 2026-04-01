import pytest
import os
import shutil
from fastapi import UploadFile
from io import BytesIO
from app.api.v1.ingest import ingest_single_file
from app.services.database.postgres_client import PostgresClient
from app.services.database.schema import CREATE_SCHEMA_QUERY 


@pytest.mark.asyncio
async def test_full_ingest_flow_integration():
    """
    Test d'intégration global de la route d'ingestion (Phase 1 & 2).
    """
    path_to_real_pdf = "app/tests/data/input/R_GBA_27_EL-GUELTA_Ayoub_MOUNASSERE_Anwar.pdf" 
    
    with open(path_to_real_pdf, "rb") as f:
        file_content = f.read()
    filename="real_test.pdf"
    fake_file = UploadFile(
        filename=filename,
        file=BytesIO(file_content),
        size=len(file_content)
    )
    db = PostgresClient()
    await db.connect()
    try:
        await db.execute(CREATE_SCHEMA_QUERY)
        await db.execute("DELETE FROM documents WHERE filename = $1", filename)

        print(f"\n🚀 Lancement de l'ingestion pour : {filename}")
        result = await ingest_single_file(fake_file)
        
        doc_id = result["doc_id"]
        assert doc_id is not None
        assert result["status"] == "success"
        assert "identity" in result
        
        # Vérification Postgres : Le document existe-t-il ?
        doc_status = await db.fetchval("SELECT status FROM documents WHERE doc_id = $1", doc_id)
        assert doc_status == "COMPLETED"
        
        # Vérification Postgres : A-t-on bien des chunks CONTENT + 1 IDENTITY ?
        content_count = await db.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND chunk_type = 'CONTENT'", doc_id
        )
        identity_count = await db.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = $1 AND chunk_type = 'IDENTITY'", doc_id
        )
        
        print(f"✅ Document {doc_id} créé avec succès.")
        print(f"✅ Chunks insérés : {content_count} Content, {identity_count} Identity.")
        
        # Vérification Stockage Physique
        expected_path = f"data/storage/{doc_id}.pdf"
        assert os.path.exists(expected_path)
        print(f"✅ Fichier stocké sur disque : {expected_path}")

    finally:
        # Nettoyage final pour laisser l'environnement propre
        # (Tu peux commenter ces lignes si tu veux aller voir les fichiers/DB après le test)
        # if 'doc_id' in locals() and os.path.exists(f"data/storage/{doc_id}.pdf"):
        #    os.remove(f"data/storage/{doc_id}.pdf")
        await db.disconnect()

if __name__ == "__main__":
    # Pour lancer le test directement : python -m pytest tests/integration/workflows/test_ingest_route.py
    pass