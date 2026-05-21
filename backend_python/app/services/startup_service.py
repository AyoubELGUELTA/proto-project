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
        count = await self.db.fetchval("SELECT COUNT(*) FROM encyclopedia")
        if count > 0:
            return
        
        # 2. Load the (mvp) json file via the repository 
        logger.info("📚 Initializing Encyclopedia from JSON source...")
        entries = self.repo.load_from_json_file() 
        
        # 3. Upsert in SQL
        for entry in entries:
            await self.repo.upsert_entry(entry)