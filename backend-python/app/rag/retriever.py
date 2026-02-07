from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from .reranker import rerank_results
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalEmbeddingClient
from ..db.postgres import get_connection, release_connection, fetch_identities_by_doc_ids 
from ..vector_store.qdrant_service import keyword_search
from collections import defaultdict
import asyncio

env = os.getenv("ENVIRONMENT", "development")
embedding_client = SolonEmbeddingClient() if env == "production" else LocalEmbeddingClient()

# Client Qdrant unique et persistant
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
client = AsyncQdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")

async def search_vector_only(query_vector, doc_id=None, collection_name="all_documents", limit=12):
    """Effectue la recherche vectorielle pure (le vecteur est déjà calculé)"""
    try:
        search_result = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=0.05,
            query_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ) if doc_id else None
        )
        return [{"chunk_id": str(hit.id), "score": hit.score} for hit in search_result.points]
    except Exception as e:
        print(f"❌ Qdrant vector search error: {repr(e)}")
        return []

async def fetch_chunks_by_ids(chunk_ids):
    if not chunk_ids: return []
    conn = await get_connection()
    query = """
        SELECT chunk_id, doc_id, chunk_index, chunk_text, chunk_visual_summary,
               chunk_heading_full, chunk_headings, chunk_tables, chunk_images_urls, created_at
        FROM chunks WHERE chunk_id = ANY($1::uuid[])
    """
    try:
        rows = await conn.fetch(query, chunk_ids)
        enriched_chunks = []
        for row in rows:
            heading = row["chunk_heading_full"]
            text_original = row["chunk_text"]
            visual_summary = row["chunk_visual_summary"] or ""
            display_text = f"### {heading}\n\n{text_original}" if heading and heading != "Sans titre" else text_original
            
            # Formatage pour le reranker
            rerank_parts = []
            if visual_summary: rerank_parts.append(f"[CONTENU VISUEL ET TABLEAUX]: {visual_summary}")
            if heading and heading != "Sans titre": rerank_parts.append(f"[TITRE/CONTEXTE]: {heading}")
            rerank_parts.append(f"[TEXTE BRUT]: {text_original}")

            enriched_chunks.append({
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": row["chunk_index"],
                "text": display_text,
                "text_for_reranker": "\n\n".join(rerank_parts),
                "visual_summary": visual_summary,
                "heading_full": heading,
                "tables": row["chunk_tables"],
                "images_urls": row["chunk_images_urls"],
                "created_at": row["created_at"]
            })
        return enriched_chunks
    finally:
        await release_connection(conn)

async def retrieve_chunks(query, doc_id=None, limit=20):
    """
    Orchestre la recherche Hybride Turbo :
    1. Lancement simultané de l'Embedding (CPU) et des Mots-clés (Qdrant).
    2. Dès que l'embedding est prêt, lancement de la recherche vectorielle.
    """
    
    # --- PHASE 1 : PARALLÉLISME DÉPART ---
    # On lance l'embedding et les mots-clés en même temps
    embedding_task = asyncio.create_task(embedding_client.embed_query(query))
    keyword_task = asyncio.create_task(keyword_search(query, limit=limit))

    # On attend les deux résultats
    # Le temps total ici = max(temps_embedding, temps_mots_clés)
    query_vector, keyword_hits = await asyncio.gather(embedding_task, keyword_task)

    # --- PHASE 2 : RECHERCHE VECTORIELLE ---
    # On a le vecteur, on peut chercher dans Qdrant
    vector_hits = await search_vector_only(query_vector, doc_id=doc_id, limit=limit)
    # --- DEBUG ---
    if keyword_hits:
        print(f"DEBUG Keyword Hit Type: {type(keyword_hits[0])}")
        print(f"DEBUG Keyword Hit Content: {keyword_hits[0]}")
# -------------
    # Fusion des IDs uniques
    combined_ids = set()
    for hit in vector_hits: combined_ids.add(hit["chunk_id"])
    for hit in keyword_hits: combined_ids.add(str(hit.id))

    if not combined_ids: return []

    print(f"🔎 Hybrid Search: {len(vector_hits)} vecteurs, {len(keyword_hits)} mots-clés. Unique: {len(combined_ids)}")

    # --- PHASE 3 : POSTGRES & RERANKER ---
    # Récupération des contenus complets
    chunks_from_db = await fetch_chunks_by_ids(list(combined_ids))
    if not chunks_from_db: return []
    
    # Reranking (Top 15 final)
    reranked_chunks = await asyncio.to_thread(
    rerank_results, 
    query, 
    chunks_from_db, 
    15
)
    # Identités des documents
    doc_ids_in_results = list({c["doc_id"] for c in reranked_chunks})
    all_identities = await fetch_identities_by_doc_ids(doc_ids_in_results)
    identity_map = {str(idnt["doc_id"]): idnt for idnt in all_identities}

    # Groupement par document pour le LLM
    grouped_by_doc = defaultdict(list)
    for chunk in reranked_chunks:
        grouped_by_doc[str(chunk["doc_id"])].append(chunk)

    final_context = []
    seen_docs = []
    for chunk in reranked_chunks:
        d_id = str(chunk["doc_id"])
        if d_id not in seen_docs:
            seen_docs.append(d_id)

    for d_id in seen_docs:
        if d_id in identity_map:
            identity = identity_map[d_id]
            identity["is_identity"] = True
            final_context.append(identity)
        for c in grouped_by_doc[d_id]:
            c["is_identity"] = False
            final_context.append(c)

    return final_context