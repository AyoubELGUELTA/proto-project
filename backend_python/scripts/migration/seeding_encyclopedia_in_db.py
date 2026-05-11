
import json
import asyncio
import argparse

from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.services.database.encyclopedia_repository import EncyclopediaRepository
from app.infrastructure.database.postgres_client import PostgresClient

import logging

logger = logging.getLogger(__name__)

async def seed_encyclopedia(json_path: str, repo: EncyclopediaRepository):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"🚀 Seeding {len(data)} entries into the SQL Encyclopedia...")

    for item in data:
        # 1. Extraction des propriétés additionnelles
        properties = {
            "aliases": item.get("ALIASES", []),
            "nasab": item.get("NASAB", ""),
            "phase": item.get("PHASE", "BOTH")
        }

        # 2. Création du modèle Pydantic
        entry = EncyclopediaEntry(
            slug=item["ID"],
            title=item["CANONICAL_NAME"],
            type=item["TYPE"],
            core_summary=item["CORE_SUMMARY"],
            properties=properties,
            is_verified=True,
            review_status="OFFICIAL"
        )

        try:
            await repo.upsert_entry(entry)
        except Exception as e:
            logger.error(f"❌ Failed to seed {item['ID']}: {e}")


    logger.info("✅ Seeding complete.")

    

async def main():
    parser = argparse.ArgumentParser(description="Seed the encyclopedia database.")
    parser.add_argument("--json", required=True, help="Path to the JSON seed file")
    args = parser.parse_args()
    
    client = PostgresClient()
    await client.connect()

    await client.initialize_schema() 

    repo = EncyclopediaRepository(client)
    
    try:
        await seed_encyclopedia(args.json, repo)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())