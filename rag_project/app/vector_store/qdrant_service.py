from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams,  Distance, PointStruct
import uuid

def store_vectors_incrementally(vectorized_docs, doc_id: str, collection_name="all_documents"):    
    """
    Store vectorized documents in a local Qdrant collection, creating it if it doesn't exist.
    
    Parameters:
        vectorized_docs (list[dict]): Each dict should have 'vector', 'content', 'metadata'
        collection_name (str): Name of the Qdrant collection
    """
    if not vectorized_docs:
        print("No documents to store.")
        return

    qdrant_client = QdrantClient(url="http://localhost:6333", timeout=60)
    
    # 1. Create collection ONCE if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        vector_size = len(vectorized_docs[0]["vector"])
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        # Create a payload index for doc_id to make filtering ultra-fast - TO LOOK AT AFTERWISE
        qdrant_client.create_payload_index(collection_name, "doc_id", field_schema="keyword")

    # 2. Upload points (they all go to the same collection)
    points = [
        PointStruct(
            id=str(uuid.uuid4()), # Use unique IDs for every chunk
            vector=doc["vector"],
            payload={
                "doc_id": doc_id,  # This is the 'Filter' key TO LOOK AT 
                "metadata": doc["metadata"]
            }
        ) for doc in vectorized_docs
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    print(f"{len(points)} documents stored successfully in collection '{collection_name}'.")
