import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from app.infrastructure.database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

class DocumentRepository:
    """
    Handles persistence and state management for documents in PostgreSQL.
    
    This repository manages the document lifecycle from initial upload (PENDING) 
    through processing (PROCESSING) to final resolution (COMPLETED/FAILED).
    """

    def __init__(self, client: PostgresClient):
        """
        Initializes the repository with a Postgres client.

        Args:
            client (PostgresClient): The database client used for execution.
        """
        self.client = client

    async def get_or_create(self, filename: str) -> str:
        """
        Registers a new document or updates the timestamp of an existing one.

        Args:
            filename (str): The original name of the uploaded file.

        Returns:
            str: The unique UUID of the document record.
        """
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
        """Transition document to the PROCESSING state (ingestion started)."""
        await self._update_status(doc_id, "PROCESSING")

    async def set_status_completed(self, doc_id: str):
        """Transition document to the COMPLETED state (pipeline finished successfully)."""
        await self._update_status(doc_id, "COMPLETED")

    async def set_status_pending(self, doc_id: str):
        """Transition document to the PENDING state (waiting in queue)."""
        await self._update_status(doc_id, "PENDING")

    async def set_status_failed(self, doc_id: str):
        """Transition document to the FAILED state (critical error encountered)."""
        await self._update_status(doc_id, "FAILED")

    async def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """
        Updates the JSONB metadata field for a specific document.
        Used to store Identity Cards or custom processing flags.

        Args:
            doc_id (str): The document identifier.
            metadata (dict): Data to be serialized into JSON.
        """
        query = "UPDATE documents SET metadata = $1 WHERE doc_id = $2"
        await self.client.execute(query, json.dumps(metadata), doc_id)

    # --- INTERNAL PRIVATE METHOD ---

    async def _update_status(self, doc_id: str, status: str):
        """
        Updates the document status and refreshes the 'updated_at' timestamp.
        """
        query = "UPDATE documents SET status = $1, updated_at = $2 WHERE doc_id = $3"
        logger.debug(f"🔄 Document {doc_id} status updated to {status}")
        await self.client.execute(query, status, datetime.now(), doc_id)