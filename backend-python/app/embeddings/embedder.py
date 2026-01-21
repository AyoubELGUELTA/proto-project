
import os
from .hf_solon_client_embedder import SolonEmbeddingClient
from .local_embedder import LocalSolonEmbeddingClient

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    embedding_client = SolonEmbeddingClient()
    print("🌐 Using Hugging Face Inference API (Production)")
else:
    embedding_client = LocalSolonEmbeddingClient()
    print("💻 Using Local MPS Embeddings (Development)")




def vectorize_documents(docs): # docs which are the summarized chunks
    """Convert LangChain Document objects into embedding vectors"""
    texts = [doc.page_content for doc in docs] #we store all the summaries of each chunk
    vectors = embedding_client.embed_documents(texts) #we embed all the list of texts to get a list of vectors
    
    vectorized_docs = []
    for chunk,vector in zip(docs,vectors):
        vectorized_docs.append({
            "chunk_id": chunk.metadata["chunk_id"], #basically the original elements: "rawtext", "tables_html", and "images_base64" # type: ignore
            "vector": vector #float numbers
        })
    print("First chunk text:", texts[0][:100], "...")  # tronqué à 100 caractères
    print("First chunk vector length:", len(vectors[0]))    
    return vectorized_docs

