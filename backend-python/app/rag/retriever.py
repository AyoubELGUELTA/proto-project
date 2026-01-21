from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalSolonEmbeddingClient
from psycopg2.extras import RealDictCursor
from ..db.postgres import get_connection

env = os.getenv("ENVIRONMENT", "development")
embedding_client = SolonEmbeddingClient() if env == "production" else LocalSolonEmbeddingClient()


def search_top_k(standalone_query, doc_id=None, collection_name="all_documents", limit=12):

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    client = QdrantClient(url=qdrant_url)   

    try:
        query_vector = embedding_client.embed_query(standalone_query)

        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.05,
            query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        ) if doc_id else None #In the future, if the user wants to select the courses he wants to ask question about
        )

        return [
            {
                "chunk_id": hit.id,
                "score": hit.score
            }
            for hit in search_result
        ]

    except Exception as e:
        print(f"❌ Qdrant search error: {repr(e)}")
        return []


def fetch_chunks_by_ids(chunk_ids):
    """
    Fetch chunks from Postgres using chunk_id UUIDs.
    """
    if not chunk_ids:
        return []

    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
        SELECT
            chunk_id,
            doc_id,
            chunk_index,
            chunk_text,
            chunk_tables,
            chunk_images_base64,
            created_at
        FROM chunks
        WHERE chunk_id = ANY(%s::uuid[])
    """

    cur.execute(query, (chunk_ids,))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": row["doc_id"],
            "chunk_index": row["chunk_index"],
            "text": row["chunk_text"],
            "tables": row["chunk_tables"],
            "images_base64": row["chunk_images_base64"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]

def retrieve_chunks(query, limit=12):
    # 1. On cherche les IDs et les scores dans Qdrant
    hits = search_top_k(query, limit=limit)
    if not hits:
        return []
    
    # 2. On extrait les IDs
    chunk_ids = [h["chunk_id"] for h in hits]

    # 3. On récupère le contenu textuel (et images/tables) dans Postgres
    chunks_from_db = fetch_chunks_by_ids(chunk_ids)

    # 4. On crée un dictionnaire pour retrouver un chunk par son ID rapidement
    chunks_by_id = {c["chunk_id"]: c for c in chunks_from_db}

    # 5. On reconstruit la liste en suivant l'ordre de Qdrant (hits)
    ordered_chunks = []
    for h in hits:
        chunk_data = chunks_by_id.get(h["chunk_id"])
        if chunk_data:
            # On injecte le score de Qdrant dans l'objet chunk
            chunk_data["vector_score"] = h["score"]
            # ICI : le texte est déjà dans chunk_data["text"] grâce à fetch_chunks_by_ids
            ordered_chunks.append(chunk_data)
    return ordered_chunks
