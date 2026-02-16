from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Document

import os

# Client partagé pour réutiliser les connexions (meilleur pour les perfs)
_client = None

def get_qdrant_client():
    global _client
    if _client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = os.getenv("QDRANT_PORT", "6333")
        _client = AsyncQdrantClient(url=f"http://{host}:{port}", timeout=60)
    return _client


async def store_vectors_incrementally(vectorized_docs, collection_name="dev_collection"):    
    """
    Store vectorized documents in Qdrant with automatic BM25 indexing.
    """
    if not vectorized_docs:
        print("⚠️ No documents to store.")
        return
    
    client = get_qdrant_client()

    # 1. Vérification et création de la collection
    try:
        exists = await client.collection_exists(collection_name)
    except Exception as e:
        print(f"❌ ERROR while checking collection: {e}")
        raise

    if not exists:
        print(f"📡 Creating collection: {collection_name}...")
        vector_size = len(vectorized_docs[0]["embedding"])
        
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, 
                distance=models.Distance.COSINE
            ),
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF  # Active le scoring BM25
                )
            }
        )
        print(f"✅ Collection créée avec sparse vectors BM25")
        
        #  Configuration de l'indexation automatique BM25 sur le champ texte
        await client.update_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(
                        on_disk=False  # En RAM pour plus de vitesse
                    )
                )
            }
        )
        print(f"✅ Index BM25 configuré")

    # 2. Préparation des points (VERSION SIMPLE)
    points = [
        models.PointStruct(
            id=doc["chunk_id"], 
            vector=doc["embedding"],  # Dense vector classique
            payload={ 
                "page_content": doc["chunk_full_content"],
                "chunk_id": doc["chunk_id"],
                # Le texte sera automatiquement index pour BM25
                "text": doc["chunk_full_content"]  
            }
        ) for doc in vectorized_docs
    ]

    # 3. Upload asynchrone
    try:
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        print(f"✅ Successfully stored {len(points)} points in {collection_name}")
    except Exception as e:
        print(f"❌ ERROR during upsert: {e}")
        raise


async def keyword_search(keywords_input, collection_name="dev_collection", limit=15):
    """
    Recherche BM25 pure (appelée par ton orchestrateur).
    """
    client = get_qdrant_client()
    query_text = " ".join(keywords_input) if isinstance(keywords_input, list) else str(keywords_input)

    try:
        # On utilise query_points mais UNIQUEMENT pour le BM25 (sparse)
        results = await client.query_points(
            collection_name=collection_name,
            query=Document(
                text=query_text, 
                model="Qdrant/bm25"
            ),
            using="bm25", 
            limit=limit,
            with_payload=True
        )
        return results.points
    except Exception as e:
        print(f"❌ BM25 Keyword search failed: {e}")
        return []







"""async def hybrid_search(query_text, embedding, collection_name="dev_collection", limit=15):
    ""
    Recherche hybride : BM25 + Semantic (dense vectors).
    Combine les résultats avec RRF (Reciprocal Rank Fusion).
    ""
    client = get_qdrant_client()
    
    print(f"🔎 Recherche hybride - Query: {query_text}")

    try:
        results = await client.query_points(
            collection_name=collection_name,
            prefetch=[
                # 1. Recherche BM25 (sparse)
                models.Prefetch(
                    query=Document(text=query_text, model="Qdrant/bm25"),
                    using="bm25",
                    limit=50  # On récupère plus de résultats pour la fusion
                ),
                # 2. Recherche sémantique (dense)
                models.Prefetch(
                    query=embedding,
                    using="",  # Dense vector par défaut
                    limit=50
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
            ),
            limit=limit
        )
        return results.points
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return []"""