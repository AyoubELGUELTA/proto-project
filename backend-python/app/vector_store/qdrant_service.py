from qdrant_client import AsyncQdrantClient, models
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
        on_disk=True,
        
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
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        
        # ✅ On configure l'index plein-texte immédiatement après création
        await setup_full_text_search(collection_name)

    # 2. Préparation des points
    points = [
        models.PointStruct(
            id=doc["chunk_id"], 
            vector=doc["embedding"],
            payload={ 
                "page_content": doc["chunk_full_content"], # Contenu pour le MatchText
                "chunk_id": doc["chunk_id"],
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
    keywords = clean_query_for_keyword(query_text)
    if not keywords:
        return []
    word_list = keywords.split()
    print(f"🔎 DEBUG Keyword Search - Query brute: '{query_text}' -> Keywords filtrés: '{keywords}'")
    results = await client.query_points(
        collection_name=collection_name,
        query=None,
        query_filter=models.Filter(
            should=[ # "should" = OR (Mieux pour le rappel)
                models.FieldCondition(
                    key="page_content", 
                    match=models.MatchText(text=word)
                ) for word in word_list
            ]
        ),
        limit=limit
    )
    return results.points


def clean_query_for_keyword(query: str) -> str:
    # Liste étendue de mots à ignorer (Stopwords français)
    stopwords = {
        "le", "la", "les", "un", "une", "des", "du", "de", "ce", "cette", "ces",
        "est", "sont", "était", "étaient", "fait", "faites", "quels", "quelles",
        "quel", "quelle", "pour", "dans", "sur", "avec", "par", "pourquoi",
        "comment", "quand", "qui", "que", "quoi", "qu'est-ce", "est-ce", "selon"
    }
    
    # Nettoyage ponctuation et mise en minuscule
    words = query.lower().replace("?", "").replace("!", "").replace(",", "").split()
    
    # On ne garde que les mots qui ne sont pas des stopwords et qui font > 2 lettres
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    
    return " ".join(filtered)