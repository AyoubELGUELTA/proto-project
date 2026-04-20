from app.infrastructure.database.postgres_client import PostgresClient
from app.services.database.encyclopedia_repository import EncyclopediaRepository

import logging

logger = logging.getLogger(__name__)

class StartupService:
    def __init__(self, db: PostgresClient, repo: EncyclopediaRepository):
        self.db = db
        self.repo = repo

    async def initialize_encyclopedia(self):
        # 1. Check if the table is empty
        count = await self.db.fetch_val("SELECT COUNT(*) FROM encyclopedia")
        if count > 0:
            return
        
        # 2. Load the (mvp) json file (FALLBACK)
        logger.info("📚 Initializing Encyclopedia from JSON source...")
        entries = self._load_json_source() 
        
        # 3. Upsert in SQL
        for entry in entries:
            await self.repo.upsert_entry(entry)