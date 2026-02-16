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
from typing import List


env = os.getenv("ENVIRONMENT", "development")
embedding_client = SolonEmbeddingClient() if env == "production" else LocalEmbeddingClient()

# Client Qdrant unique et persistant
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = os.getenv("QDRANT_PORT", "6333")
client = AsyncQdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")

async def search_vector_only(query_vector, doc_id=None, collection_name="dev_collection", limit=12):
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
        SELECT chunk_id, doc_id, chunk_index, chunk_text, chunk_page_numbers, chunk_visual_summary,
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
            
            # Formatage pour le reranker
            rerank_parts = []
            if visual_summary: rerank_parts.append(f"[CONTENU VISUEL ET TABLEAUX]: {visual_summary}")
            if heading and heading != "Sans titre": rerank_parts.append(f"[TITRE/CONTEXTE]: {heading}")
            rerank_parts.append(f"[TEXTE BRUT]: {text_original}")

            enriched_chunks.append({
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_index": row["chunk_index"],
                "text_for_reranker": "\n\n".join(rerank_parts),
                "page_numbers": row["chunk_page_numbers"],
                "visual_summary": visual_summary,
                "heading_full": heading,
                "tables": row["chunk_tables"],
                "images_urls": row["chunk_images_urls"],
                "created_at": row["created_at"]
            })
        return enriched_chunks
    finally:
        await release_connection(conn)

async def retrieve_chunks(query_data, rerank_limit=20 , doc_id=None, limit=50):
    """
    Orchestre la recherche Hybride Turbo :
    1. Lancement simultané de l'Embedding (CPU) et des Mots-clés (Qdrant).
    2. Dès que l'embedding est prêt, lancement de la recherche vectorielle.
    """
    
    # 1. Génération des vecteurs 
    # Pour l'exemple, on utilise la query principale et on parallélise
    
    print("🚀 Démarrage Hybrid RRF Retrieval...")
    
    variants = query_data.get("variants", [query_data["vector_query"]])

    # Tâches asynchrones
    embedding_tasks = [asyncio.create_task(embedding_client.embed_query(v)) for v in variants]
    keyword_task = asyncio.create_task(keyword_search(query_data["keyword_query"], limit=limit))
    
    # On attend les vecteurs
    vectors = await asyncio.gather(*embedding_tasks)
    keyword_hits = await keyword_task
    
    # 2. Recherches Qdrant en parallèle
    # On pourrait ici avoir plusieurs vecteurs si on avait fait du Multi-Query
    search_tasks = [
        asyncio.create_task(search_vector_only(vec, doc_id=doc_id, limit=limit)) 
        for vec in vectors
    ]
    all_vector_results = await asyncio.gather(*search_tasks)
    
    # PHASE 3 : RRF (Fusion des rangs)
    all_rankings = []
    # Ajouter les classements vectoriels
    for search_result in all_vector_results:
        all_rankings.append([hit["chunk_id"] for hit in search_result])
    # Ajouter le classement BM25
    all_rankings.append([str(hit.id) for hit in keyword_hits])

    combined_ids_sorted = compute_rrf(all_rankings)
    
    # PHASE 4 : Fetch & Rerank
    top_ids = combined_ids_sorted[:limit]
    chunks_from_db = await fetch_chunks_by_ids(top_ids)
    
    reranked_chunks = await asyncio.to_thread(
        rerank_results, 
        query_data["vector_query"], # On rerank toujours sur la query originale
        chunks_from_db, 
        top_n=rerank_limit
    )

    return await finalize_context(reranked_chunks)

async def finalize_context(reranked_chunks):
    if not reranked_chunks:
        return []

    # 1. Récupérer les identités
    doc_ids_in_results = list({c["doc_id"] for c in reranked_chunks})
    all_identities = await fetch_identities_by_doc_ids(doc_ids_in_results)
    identity_map = {str(idnt["doc_id"]): idnt for idnt in all_identities}

    # 2. Groupement par document ET stockage du meilleur score par doc
    grouped_by_doc = defaultdict(list)
    doc_best_score = {}

    for chunk in reranked_chunks:
        d_id = str(chunk["doc_id"])
        grouped_by_doc[d_id].append(chunk)
        
        # On garde une trace du score le plus élevé pour ce document
        current_score = chunk.get("rerank_score", 0)
        if d_id not in doc_best_score or current_score > doc_best_score[d_id]:
            doc_best_score[d_id] = current_score

    # 3. Trier les identités de documents par leur meilleur score de chunk
    # (Comme ça, le livre le plus pertinent arrive toujours en premier)
    sorted_doc_ids = sorted(doc_best_score.keys(), key=lambda x: doc_best_score[x], reverse=True)

    # 4. Reconstruction propre du contexte
    final_context = []
    for d_id in sorted_doc_ids:
        # A. Ajouter l'identité du doc (une seule fois)
        if d_id in identity_map:
            identity = identity_map[d_id].copy()
            identity["is_identity"] = True
            final_context.append(identity)
        
        # B. Ajouter TOUS les chunks de ce document (triés par index pour la logique du récit)
        # On trie par 'chunk_index' pour que l'histoire se suive
        chunks_of_this_doc = sorted(grouped_by_doc[d_id], key=lambda x: x.get("chunk_index", 0))
        
        for c in chunks_of_this_doc:
            chunk_to_add = c.copy()
            chunk_to_add["is_identity"] = False
            final_context.append(chunk_to_add)

    return final_context

def compute_rrf(rankings: List[List[str]], k=60):
    """
    rankings: Liste de listes d'IDs (chaque sous-liste est un classement d'une recherche)
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            # rank + 1 car l'index commence à 0
            scores[chunk_id] += 1 / (k + (rank + 1))
    
    # Trier par score décroissant
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_ids]