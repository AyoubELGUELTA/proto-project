from .base import release_connection, get_connection
from typing import List, Dict, Any, Optional


async def get_or_create_document(filename: str) -> str:
    """Récupère ou crée un document"""
    conn = await get_connection()
    try:
        # ✅ $1 au lieu de %s
        doc_id = await conn.fetchval("""
            INSERT INTO documents (filename) 
            VALUES ($1) 
            ON CONFLICT (filename) DO UPDATE SET filename = EXCLUDED.filename
            RETURNING doc_id;
        """, filename)
        return str(doc_id)
    finally:
        await release_connection(conn)


async def get_documents() -> List[str]:
    """Récupère la liste des documents"""
    conn = await get_connection()
    try:
        rows = await conn.fetch(
            "SELECT filename FROM documents ORDER BY created_at DESC;"
        )
        return [row['filename'] for row in rows]
    finally:
        await release_connection(conn)