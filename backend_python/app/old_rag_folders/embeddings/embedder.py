
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



async def vectorize_documents(docs):
    vectorized_docs = []

    for chunk in docs:
        heading = chunk.get("heading_full", "")
        text = chunk.get("text", "")
        visual = chunk.get("visual_summary", "")
        
        # 1. On prépare le texte COMPLET pour l'embedding (Titre + Texte + Visuel)
        # C'est ce mélange qui donne la meilleure "empreinte" sémantique
        parts = []
        if heading and heading != "Sans titre":
            parts.append(f"# {heading}")
        parts.append(text)
        if visual:
            parts.append(f"[VISUAL DESCRIPTION] {visual}")
            
        full_string_to_embed = "\n\n".join(parts)

        # 2. Appel à l'embedding avec la version enrichie
        embedding = await embedding_client.embed_documents(full_string_to_embed)
        
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
        
        # 3. Stockage Qdrant
        vectorized_docs.append({
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding,
            "chunk_full_content": full_string_to_embed # On stocke la version complète
        })

    return vectorized_docs