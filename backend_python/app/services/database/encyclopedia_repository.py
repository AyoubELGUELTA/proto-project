import logging
import json
from typing import Optional, List
from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.infrastructure.database.postgres_client import PostgresClient
import os


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
        
    
    def load_from_json_file(self, file_path: str = "app/core/resources/encyclopedia_fallback.json") -> List[EncyclopediaEntry]:
        """
        Loads the fallback initialization entries from a local JSON file.
        Supports both the legacy MVP uppercase format and the new structured model format.
        """
        import os
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Encyclopedia fallback file not found at {file_path}. Returning empty list.")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            entries = []
            for item in raw_data:
                # 1. SI LE FICHIER EST DÉJÀ AU NOUVEAU FORMAT (Minuscules)
                if "title" in item or "core_summary" in item:
                    if isinstance(item.get("properties"), str):
                        item["properties"] = json.loads(item["properties"])
                    entries.append(EncyclopediaEntry(**item))
                    continue
                
                # 2. ADAPTER DE COMPATIBILITÉ POUR LE FORMAT MVP (Majuscules)
                # On package les champs spécifiques ('ALIASES', 'NASAB', 'PHASE') dans le dict 'properties'
                mvp_properties = {
                    "aliases": item.get("ALIASES", []),
                    "nasab": item.get("NASAB"),
                    "phase": item.get("PHASE")
                }
                # On nettoie les clés qui vaudraient None pour garder un dictionnaire propre
                mvp_properties = {k: v for k, v in mvp_properties.items() if v is not None}

                mapped_data = {
                    # On garde l'ID sémantique (ex: 'UMAR_IBN_AL_KHATTAB') au lieu de générer un UUID random,
                    # c'est beaucoup plus propre pour une encyclopédie fixe de référence !
                    "id": item.get("ID"), 
                    "title": item.get("CANONICAL_NAME"),
                    "type": item.get("TYPE"),
                    "core_summary": item.get("CORE_SUMMARY"),
                    "properties": mvp_properties,
                    "review_status": "OFFICIAL", # Valeur par défaut
                    "is_verified": True
                }
                
                # Validation Pydantic transparente
                entries.append(EncyclopediaEntry(**mapped_data))
            
            logger.info(f"📚 Successfully loaded {len(entries)} entries (MVP format dynamically adapted).")
            return entries
            
        except Exception as e:
            logger.error(f"❌ Failed to parse encyclopedia fallback JSON file: {e}")
            return []