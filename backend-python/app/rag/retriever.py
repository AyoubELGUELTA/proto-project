from qdrant_client import QdrantClient, models
import os
from ..embeddings.hf_solon_client_embedder import SolonEmbeddingClient
from ..embeddings.local_embedder import LocalSolonEmbeddingClient

env = os.getenv("ENVIRONMENT", "development")
embedding_client = SolonEmbeddingClient() if env == "production" else LocalSolonEmbeddingClient()


def search_top_k(standalone_query, doc_id=None, collection_name="all_documents", limit=12):

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    client = QdrantClient(url=qdrant_url)   
    try:
        # 2. Vectorisation de la requête
        # Note: On garde le préfixe "query: " car Solon en a besoin pour la recherche
        query_vector = embedding_client.embed_query(standalone_query)
        
        # 3. Construction du filtre
        search_filter = None
        if doc_id:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", 
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )

        # 4. Recherche
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "score": hit.score, 
                "metadata": hit.payload.get("metadata"), 
                "doc_id": hit.payload.get("doc_id")
            }
            for hit in search_result.points
        ]

    except Exception as e:
        print(f"❌ Erreur lors de la recherche Qdrant: {e}")
        return []