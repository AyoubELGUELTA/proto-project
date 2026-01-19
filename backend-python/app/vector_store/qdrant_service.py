from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams,  Distance, PointStruct
import os
import uuid

def get_qdrant_client():
    """
    Initializes the Qdrant client using environment variables    
    """
    
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    url = f"http://{host}:{port}"
    
    api_key = os.getenv("QDRANT_API_KEY") # to look at in prod 
    
    return QdrantClient(url=url, api_key=api_key, timeout=60)

def store_vectors_incrementally(vectorized_docs, collection_name="all_documents"):    
    """
    Store vectorized documents in a local Qdrant collection, creating it if it doesn't exist.
    
    Parameters:
        vectorized_docs (list[dict]): Each dict should have 'vector', 'content', 'metadata'
        collection_name (str): Name of the Qdrant collection
    """
    if not vectorized_docs:
        print("No documents to store.")
        return

    qdrant_client = get_qdrant_client()
    
    # 1. Create collection ONCE if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        vector_size = len(vectorized_docs[0]["vector"])
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )


    # 2. Upload points (they all go to the same collection)
    points = [
        PointStruct(
            id=doc["chunk_id"], # Use unique IDs for every chunk
            vector=doc["vector"],
            payload={} # WE ACCESS THE PAYLOAD FROM POSTGRES, WE ONLY STORE VECTORES + CHUNK IDS IN QDRANT
        ) for doc in vectorized_docs
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    print(f"{len(points)} documents stored successfully in collection '{collection_name}'.")
