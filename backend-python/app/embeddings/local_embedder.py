import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List

class LocalEmbeddingClient:
    def __init__(self):
        # "mps" tells the Mac to use its Metal GPU (Silicon chip)
        # If MPS isn't available, it defaults to CPU
        # Dans Docker sur Mac, MPS n'est souvent pas exposé, on forcera donc le CPU si besoin

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"🚀 Initializing BGE-M3 locally on: {self.device}")

        model_name = os.getenv("LOCAL_EMBEDDER_NAME", "BAAI/bge-m3")

        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=True  # ✅ Nécessaire pour BGE-M3
        )
        print(f"✅ BGE-M3 loaded successfully (max tokens: 8192)")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embedde les documents/chunks pour l'indexation.
        BGE-M3 n'a pas besoin de préfixe spécifique pour les passages.
        """
        if isinstance(texts, str):
            texts = [texts]  # Need to be a list of str

        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=32,  # ✅ Ajuster selon votre RAM/GPU
            show_progress_bar=True  # ✅ Utile pour debug
        )
        
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        # Solon performs better if you tell it it's a query
        instructional_query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            instructional_query, 
            convert_to_numpy=True

        ) 
        return embedding.tolist()