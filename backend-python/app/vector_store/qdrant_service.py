from qdrant_client import AsyncQdrantClient 
from qdrant_client.http import models
from qdrant_client.models import PointStruct, VectorParams, Distance
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

async def setup_full_text_search(collection_name):
    client = get_qdrant_client()
    await client.create_payload_index(
    collection_name=collection_name,
    field_name="page_content",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.MULTILINGUAL,
        lowercase=True,
        ascii_folding=True, # Très important pour les recherches en français/arabe
        # On regroupe les paramètres techniques dans index_params
        index_params=models.TextIndexConfig(
            on_disk=True,
            distance=1 # Fuzzy Match
        )
    )
)


async def store_vectors_incrementally(vectorized_docs, collection_name="all_documents"):    
    """
    Store vectorized documents in Qdrant (Async version).
    """
    if not vectorized_docs:
        print("⚠️ No documents to store.")
        return
    
    client = get_qdrant_client()

    # 1. Vérification et création de la collection (Async)
    try:
        exists = await client.collection_exists(collection_name)
    except Exception as e:
        print(f"❌ ERROR while checking collection: {e}")
        raise

    if not exists:
        print(f"📡 Creating collection: {collection_name}...")
        # On récupère la taille du premier embedding pour configurer la collection
        vector_size = len(vectorized_docs[0]["embedding"])
        
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # ✅ On configure l'index plein-texte immédiatement après création
        await setup_full_text_search(collection_name)

    # 2. Préparation des points
    points = [
        PointStruct(
            id=doc["chunk_id"], 
            vector=doc["embedding"],
            payload={ 
                "page_content": doc["chunk_full_content"], # Contenu pour le MatchText
                "chunk_id": doc["chunk_id"],
                "doc_id": doc["metadata"].get("doc_id") # Important pour tes filtres par doc !
            }
        ) for doc in vectorized_docs
    ]

    # 3. Upload (Upsert) Asynchrone
    try:
        await client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True # On attend que l'indexation soit finie pour confirmer
        )
        print(f"✅ Successfully stored {len(points)} points in {collection_name}")
    except Exception as e:
        print(f"❌ ERROR during upsert: {e}")


async def keyword_search(query_text, collection_name="all_documents", limit=15):
    client = get_qdrant_client()
    # On nettoie un peu la query pour le plein texte
    keywords = " ".join([w for w in query_text.split() if len(w) > 2])
    
    results = await client.query_points( 
        collection_name=collection_name,
        query=None,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="page_content", match=models.MatchText(text=keywords))]
        ),
        limit=limit
    )
    return results.points

