from qdrant_client import QdrantClient, models
import os
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalSolonEmbeddingClient
from psycopg2.extras import RealDictCursor

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
            limit=limit
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
        WHERE chunk_id = ANY(%s)
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
    hits = search_top_k(query, limit=limit)

    chunk_ids = [h["chunk_id"] for h in hits]
    scores = {h["chunk_id"]: h["score"] for h in hits}

    chunks = fetch_chunks_by_ids(chunk_ids)

    for chunk in chunks:
        chunk["vector_score"] = scores.get(chunk["chunk_id"], 0)

    chunks.sort(key=lambda x: x["vector_score"], reverse=True)
    return chunks
