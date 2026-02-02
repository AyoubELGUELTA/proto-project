
import os
from .hf_solon_client_embedder import SolonEmbeddingClient
from .local_embedder import LocalEmbeddingClient

env = os.getenv("ENVIRONMENT", "development")

if env == "production":
    embedding_client = SolonEmbeddingClient()
    print("🌐 Using Hugging Face Inference API (Production)")
else:
    embedding_client = LocalEmbeddingClient()
    print("💻 Using Local MPS Embeddings (Development)")




def vectorize_documents(docs): # docs which are the summarized chunks
    """Convert LangChain Document objects into embedding vectors"""

    vectorized_docs = []


    for chunk in docs:
        # ✅ Construire le texte enrichi avec le titre
        heading = chunk.get("heading_full", "")
        text = chunk.get("text", "")
        
        if heading and heading != "Sans titre":
            # Ajouter le titre au début du texte pour l'embedding
            enriched_text = f"# {heading}\n\n{text}"
        else:
            enriched_text = text

        embedding = embedding_client.embed_documents(enriched_text)
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]  # Extraire le vecteur réel
        
        
        vectorized_docs.append({
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding
        })

    print(f"\n🔍 DEBUG embedding:")
    print(f"  Type: {type(vectorized_docs[0]['embedding'])}")    

    print("First chunk vector length:", len(vectorized_docs[0]))    
    return vectorized_docs

