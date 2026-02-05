from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from .reranker import rerank_results
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalEmbeddingClient
from ..db.postgres import get_connection, release_connection, fetch_identities_by_doc_ids 
import asyncio

env = os.getenv("ENVIRONMENT", "development")
embedding_client = SolonEmbeddingClient() if env == "production" else LocalEmbeddingClient()


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


async def fetch_chunks_by_ids(chunk_ids):
    """
    Fetch chunks from Postgres using chunk_id UUIDs.
    """
    if not chunk_ids:
        return []

    conn = await get_connection()
    query = """
        SELECT
            chunk_id,
            doc_id,
            chunk_index,
            chunk_text,
            chunk_visual_summary,
            chunk_heading_full,
            chunk_headings,
            chunk_tables,
            chunk_images_urls,
            created_at
        FROM chunks
        WHERE chunk_id = ANY($1::uuid[])
    """

    try:
        # asyncpg utilise $1, $2 au lieu de %s et fetch() renvoie des records type dict
        rows = await conn.fetch(query, chunk_ids)
        
        enriched_chunks = []
        for row in rows:
            heading = row["chunk_heading_full"]
            text_original = row["chunk_text"]
            visual_summary = row["chunk_visual_summary"] or ""

            display_text = f"### {heading}\n\n{text_original}" if heading and heading != "Sans titre" else text_original
            

            rerank_parts = []

            if visual_summary:
                rerank_parts.append(f"[CONTENU VISUEL ET TABLEAUX]: {visual_summary}")
            
            if heading and heading != "Sans titre":
                rerank_parts.append(f"[TITRE/CONTEXTE]: {heading}")
            
            rerank_parts.append(f"[TEXTE BRUT]: {text_original}")

            rerank_text = "\n\n".join(rerank_parts)

            enriched_chunks.append({
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": row["chunk_index"],
                "text": display_text,
                "text_for_reranker": rerank_text,
                "visual_summary": visual_summary,
                "heading_full": heading,
                "headings": row["chunk_headings"],
                "tables": row["chunk_tables"],
                "images_urls": row["chunk_images_urls"], # URLs S3
                "created_at": row["created_at"]
            })
        return enriched_chunks
    
    except Exception as e: 
        print(f"Error while fetching chunks in the retrieving: {e}")
        return 
    finally:
        await release_connection(conn)

async def retrieve_chunks(query, doc_id=None, limit=30):
    # 1. Qdrant
    hits = search_top_k(query, doc_id=doc_id, limit=limit)
    if not hits: return []
    
    # 2. Postgres
    chunks_from_db = await fetch_chunks_by_ids([h["chunk_id"] for h in hits])
    if not chunks_from_db: return []
    
    # 3. Reranking (On en garde 15 pour être large)
    reranked_chunks = rerank_results(query, chunks_from_db, top_n=15)
    
    # 4. Récupération des Identités
    doc_ids_in_results = list({c["doc_id"] for c in reranked_chunks})
    all_identities = await fetch_identities_by_doc_ids(doc_ids_in_results)
    
    # Création d'un dictionnaire pour accès rapide aux identités {doc_id: identity_chunk}
    identity_map = {str(idnt["doc_id"]): idnt for idnt in all_identities}
    
    final_context = []
    seen_doc_identities = set()

    for chunk in reranked_chunks:
        curr_doc_id = str(chunk["doc_id"])
        
        # SI c'est le premier chunk qu'on voit pour ce document, 
        # on insère l'identité du doc JUSTE AVANT
        if curr_doc_id not in seen_doc_identities:
            if curr_doc_id in identity_map:
                # On marque l'identité pour que le front sache faire la différence
                identity = identity_map[curr_doc_id]
                identity["is_identity"] = True 
                final_context.append(identity)
            seen_doc_identities.add(curr_doc_id)
        
        # On ajoute le chunk technique
        chunk["is_identity"] = False
        final_context.append(chunk)

    print(f"DEBUG: Nombre de sources envoyées au front: {len(final_context)}")
    # Tu devrais voir ici un chiffre comme 9 (1 identité + 8 chunks) ou 16 (1 identité + 15 chunks)
    
    return final_context
