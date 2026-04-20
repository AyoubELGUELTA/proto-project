import logging
from typing import Optional, List
from app.core.data_model.encyclopedia import EncyclopediaEntry
from infrastructure.database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

class EncyclopediaRepository:

    def __init__(self, client: PostgresClient):
        self.client = client

    async def get_by_id(self, entity_id: str) -> Optional[EncyclopediaEntry]:
        """Fetches a single encyclopedia entry by its unique ID."""
        try:
            query = "SELECT * FROM encyclopedia WHERE id = $1"
            row = await self.client.fetch_row(query, entity_id)
            return EncyclopediaEntry(**dict(row)) if row else None
        except Exception as e:
            logger.error(f"❌ Error fetching entity {entity_id} from encyclopedia: {e}")
            return None

    async def upsert_entry(self, entry: EncyclopediaEntry):
        """Inserts a new entry or updates an existing one in the encyclopedia."""

        try:
            data = entry.model_dump()

            query = """
            INSERT INTO encyclopedia (id, slug, title, type, category, core_summary, properties, is_verified, review_status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE SET
                slug = EXCLUDED.slug,
                core_summary = EXCLUDED.core_summary,
                properties = encyclopedia.properties || EXCLUDED.properties,
                updated_at = CURRENT_TIMESTAMP
                category = EXCLUDED.category, -- If the category change
            """
            await self.client.execute(
            query, data['id'], data['slug'], data['title'], data['type'], 
            data['category'], data['core_summary'], data['properties'], 
            data['is_verified'], data['review_status']
        )
        except Exception as e:
            logger.error(f"❌ Error upserting entry {entry.id}: {e}")
            raise

    async def search_by_criteria(self, slug: str, category: str) -> List[EncyclopediaEntry]:
        """
        Searches encyclopedia entries based on category and fuzzy/exact matching.

        Logic: Category check, ID match, Alias overlap, and partial containment.
        """
        try:
            query = """
            SELECT * FROM encyclopedia 
            WHERE category = $2 
            AND (
                slug = $1 
                OR properties->'aliases' @> jsonb_build_array($1)
                OR (length($1) > 4 AND (
                    slug LIKE '%' || $1 || '%' 
                    OR EXISTS (
                        SELECT 1 FROM jsonb_array_elements_text(properties->'aliases') AS a 
                        WHERE a LIKE '%' || $1 || '%'
                    )
                ))
            )
            """
            rows = await self.client.fetch(query, slug, category)
            return [EncyclopediaEntry(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"❌ Error searching encyclopedia for {slug} (cat: {category}): {e}")
            return []