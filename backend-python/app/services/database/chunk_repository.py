from typing import List
from app.models.domain import TextUnit
import json

class ChunkRepository:
    def __init__(self, client):
        self.client = client

    async def store_text_units(self, doc_id: str, units: List[TextUnit], chunk_type: str = "CONTENT"):
        """
        Insère un batch de TextUnits dans PostgreSQL.
        """
        records = []
        for i, unit in enumerate(units):
            records.append((
                doc_id,
                i,
                chunk_type,
                unit.text,
                json.dumps(unit.headings) if unit.headings else '[]',
                unit.metadata.get("heading_full", ""),
                unit.page_numbers if unit.page_numbers else [], 
                json.dumps(unit.tables) if unit.tables else '[]',
                list(unit.metadata.get("image_urls", [])),
                json.dumps(unit.metadata)
            ))

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