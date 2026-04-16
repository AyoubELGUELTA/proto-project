import json
import logging
from typing import List
from app.models.domain import TextUnit

logger = logging.getLogger(__name__)

class ChunkRepository:
    """
    Handles bulk persistence of TextUnits into the PostgreSQL 'chunks' table.
    
    Optimized for performance using binary copy protocols for high-volume 
    data insertion during document ingestion.
    """

    def __init__(self, client):
        """
        Initializes the repository with a database client.

        Args:
            client (PostgresClient): The underlying database client with pool access.
        """
        self.client = client

    async def store_text_units(self, doc_id: str, units: List[TextUnit], chunk_type: str = "CONTENT"):
        """
        Inserts a batch of TextUnits into PostgreSQL using the COPY protocol.

        Args:
            doc_id (str): The ID of the parent document.
            units (List[TextUnit]): The list of processed fragments to store.
            chunk_type (str): The classification of the chunk (e.g., 'CONTENT', 'IDENTITY').
        """
        if not units:
            logger.debug(f"⚠️ No units provided for document {doc_id}. Skipping database storage.")
            return

        records = []
        for i, unit in enumerate(units):
            # Prepare the tuple for the COPY command
            records.append((
                doc_id,
                i,                          # chunk_index
                chunk_type,
                unit.text,
                json.dumps(unit.headings) if unit.headings else '[]',
                unit.metadata.get("heading_full", ""),
                unit.page_numbers if unit.page_numbers else [], 
                json.dumps(unit.tables) if unit.tables else '[]',
                list(unit.metadata.get("image_urls", [])),
                json.dumps(unit.metadata)
            ))

        try:
            # We use the internal pool to acquire a connection for the COPY operation
            async with self.client._pool.acquire() as conn:
                await conn.copy_records_to_table(
                    'chunks',
                    records=records,
                    columns=[
                        'doc_id', 'chunk_index', 'chunk_type', 'chunk_text', 
                        'chunk_headings', 'chunk_heading_full', 'chunk_page_numbers', 
                        'chunk_tables', 'chunk_images_urls', 'chunk_metadata'
                    ]
                )
            logger.info(f"💾 Successfully stored {len(units)} units for document {doc_id} in 'chunks' table.")
            
        except Exception as e:
            logger.error(f"❌ Failed to bulk store chunks for doc {doc_id}: {e}")
            raise e