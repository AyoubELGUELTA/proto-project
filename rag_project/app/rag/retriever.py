from qdrant_client import QdrantClient
import os
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalSolonEmbeddingClient

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    embedding_client = SolonEmbeddingClient()
    print("🌐 Using Hugging Face Inference API (Production)")
else:
    embedding_client = LocalSolonEmbeddingClient()
    print("💻 Using Local MPS Embeddings (Development)")


from qdrant_client import models

def search_top_k(query, doc_id=None, collection_name="all_documents", limit=5):
    client = QdrantClient(url="http://localhost:6333")
    query_vector = embedding_client.embed_query(query)
    
    # Build the filter if a doc_id is provided
    search_filter = None
    if doc_id:
        search_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ) # TO LOOK AT AT THE END

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=search_filter, # If None, it searches EVERYTHING
        limit=limit,
        with_payload=True
    )
    
    return [
        {"score": hit.score, "text": hit.payload.get("metadata"), "doc_id": hit.payload.get("doc_id")}
        for hit in search_result.points
    ]