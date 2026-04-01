# app/services/database/document_repository.py
import json
from datetime import datetime
from .postgres_client import PostgresClient

class DocumentRepository:
    def __init__(self, client: PostgresClient):
        self.client = client

    async def get_or_create(self, filename: str) -> str:
        query = """
            INSERT INTO documents (filename, updated_at, status) 
            VALUES ($1, $2, 'PENDING') 
            ON CONFLICT (filename) 
            DO UPDATE SET updated_at = $2
            RETURNING doc_id;
        """
        doc_id = await self.client.fetchval(query, filename, datetime.now())
        return str(doc_id)


    async def set_status_processing(self, doc_id: str):
        """Marque le début du travail lourd."""
        await self._update_status(doc_id, "PROCESSING")

    async def set_status_completed(self, doc_id: str):
        """Marque la réussite totale de l'ingestion."""
        await self._update_status(doc_id, "COMPLETED")

    async def set_status_pending(self, doc_id: str):
        """Met en suspend l'ingestion du document."""
        await self._update_status(doc_id, "PENDING")

    async def set_status_failed(self, doc_id: str):
        """Marque un échec critique."""
        await self._update_status(doc_id, "FAILED")

    # --- MÉTHODE INTERNE PRIVÉE ---

    async def _update_status(self, doc_id: str, status: str):
        """Seule méthode qui touche vraiment au champ status en SQL."""
        query = "UPDATE documents SET status = $1, updated_at = $2 WHERE doc_id = $3"
        await self.client.execute(query, status, datetime.now(), doc_id)

    async def update_metadata(self, doc_id: str, metadata: dict):
        query = "UPDATE documents SET metadata = $1 WHERE doc_id = $2"
        await self.client.execute(query, json.dumps(metadata), doc_id)