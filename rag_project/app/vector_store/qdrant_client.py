from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams,  Distance, PointStruct

def store_vectors_incrementally(vectorized_docs, collection_name="ai_summary_chunks"):
    """
    Store vectorized documents in a local Qdrant collection, creating it if it doesn't exist.
    
    Parameters:
        vectorized_docs (list[dict]): Each dict should have 'vector', 'content', 'metadata'
        collection_name (str): Name of the Qdrant collection
    """
    if not vectorized_docs:
        print("No documents to store.")
        return

    qdrant_client = QdrantClient(url="http://localhost:6333")
    
    # Check if collection exists
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        # Create new collection with vector size from first document
        vector_size = len(vectorized_docs[0]["vector"])
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance = Distance.COSINE)
        )
    else:
        print(f"Collection '{collection_name}' exists. Adding new vectors...")

    # Prepare points
    start_id = qdrant_client.count(collection_name=collection_name).count
    points = [
    PointStruct(
        id=start_id + i,
        vector=doc["vector"],
        payload={
            "metadata": doc["metadata"]  # the original raw elements, text, table as html_text, and image as base64
        }
    )
    for i, doc in enumerate(vectorized_docs)
]

    # Upsert points into Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"{len(points)} documents stored successfully in collection '{collection_name}'.")
