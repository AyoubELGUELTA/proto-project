import logging
import json
from typing import Optional, List
from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.infrastructure.database.postgres_client import PostgresClient

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
            properties_json = json.dumps(data['properties']) #propreties filed is an str field (cf schema)

            query = """
            INSERT INTO encyclopedia (id, slug, title, type, category, core_summary, properties, is_verified, review_status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (slug) DO UPDATE SET -- We merge on the slug which is unique
            core_summary = EXCLUDED.core_summary,
            properties = encyclopedia.properties || EXCLUDED.properties,
            category = EXCLUDED.category, 
            updated_at = CURRENT_TIMESTAMP
            """
            await self.client.execute(
            query, data['id'], data['slug'], data['title'], data['type'], 
            data['category'], data['core_summary'], properties_json, 
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
                slugify_entity(slug) = $1

                OR EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(properties->'aliases') AS a
                    WHERE slugify_entity(a) = $1
                )

                OR (
                    length($1) > 4
                    AND (
                        slugify_entity(slug) LIKE '%' || $1 || '%'

                        OR EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements_text(properties->'aliases') AS a
                            WHERE slugify_entity(a) LIKE '%' || $1 || '%'
                        )
                    )
                )
            )
            """
            logger.info(f"🧪 SEARCH ENCYCLO QUERY slug={slug} category={category}")
            rows = await self.client.fetch(query, slug, category)
            logger.info(f"📦 RESULTS COUNT: {len(rows)}")   
            results = []

            for r in rows:
                logger.info(f"➡️ MATCH: id={r['id']} slug={r['slug']} title={r['title']}")

                data = dict(r)

                # FIX UUID -> string for the Model
                data["id"] = str(data["id"])

                # FIX JSONB string -> dict (for the Model)
                if isinstance(data.get("properties"), str):

                    data["properties"] = json.loads(data["properties"])

                results.append(EncyclopediaEntry(**data))

            return results

        except Exception as e:
            logger.error(f"❌ Error searching encyclopedia for {slug} (cat: {category}): {e}")
            return []