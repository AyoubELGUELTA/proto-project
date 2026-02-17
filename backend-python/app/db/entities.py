from .base import get_connection, release_connection
import uuid
from typing import List, Dict, Any, Optional

async def resolve_entity(extracted_name: str, extracted_aliases: List[str], conn=None):
    """
    Cherche la meilleure entité. 
    Acceptant 'conn' pour être utilisé à l'intérieur d'une transaction existante.
    """
    should_release = False
    if conn is None:
        conn = await get_connection()
        should_release = True

    try:
        all_extracted = set([extracted_name] + extracted_aliases)
        
        # On récupère les candidats (ID, Name, Aliases)
        candidates = await conn.fetch("""
            SELECT entity_id, name, aliases FROM entities 
            WHERE name = ANY($1::text[]) 
               OR aliases && $1::text[]
        """, list(all_extracted))

        if not candidates:
            return None

        best_cand = None
        max_score = -1

        for cand in candidates:
            cand_names = set([cand['name']] + (cand['aliases'] or []))
            score = len(all_extracted.intersection(cand_names))
            if score > max_score:
                max_score = score
                best_cand = cand # On retourne l'objet complet pour avoir les aliases plus tard

        return best_cand
    finally:
        if should_release:
            await release_connection(conn)

async def link_entity_to_chunk(chunk_id: str, extracted_entity: Dict[str, Any]):
    conn = await get_connection()
    name = extracted_entity['name']
    aliases = extracted_entity.get('aliases', [])
    entity_type = extracted_entity.get('type', 'CONCEPT')

    try:
        async with conn.transaction():
            # Passation de la connexion 'conn' pour rester dans la transaction, if conn == true, pas besoin de se reco, et vice-versa...
            entity = await resolve_entity(name, aliases, conn=conn)

            if entity:
                entity_id = entity['entity_id']
                existing_aliases = set(entity['aliases'] or [])
                new_aliases = existing_aliases.union(aliases)
                if len(new_aliases) > len(existing_aliases):
                    await conn.execute(
                        "UPDATE entities SET aliases = $1 WHERE entity_id = $2",
                        list(new_aliases), entity_id
                    )
            else:
                entity_id = await conn.fetchval("""
                    INSERT INTO entities (name, aliases, entity_type)
                    VALUES ($1, $2, $3)
                    RETURNING entity_id;
                """, name, aliases, entity_type)

            await conn.execute("""
                INSERT INTO entity_links (entity_id, chunk_id, relevance_score)
                VALUES ($1, $2, $3)
                ON CONFLICT (entity_id, chunk_id) DO NOTHING;
            """, entity_id, uuid.UUID(chunk_id), extracted_entity.get('relevance', 1.0))
    except Exception as e:
        print(f"❌ Erreur link_entity: {e}")
        raise
    finally:
        await release_connection(conn)