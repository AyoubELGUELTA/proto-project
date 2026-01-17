import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List

class LocalSolonEmbeddingClient:
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
        print(f"🚀 Initializing Solon locally on: {self.device}")

        model_name = os.getenv("LOCAL_EMBEDDER_NAME", "OrdalieTech/Solon-embeddings-large-0.1")

        self.model = SentenceTransformer(
            model_name, 
            device=self.device
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        
        if isinstance(texts, str):
            texts = [texts] # Need to by a list of str

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Manually convert the resulting NumPy array to a list, otherwise it throws an error..
        
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        # Solon performs better if you tell it it's a query
        instructional_query = f"query: {query}"
        embedding = self.model.encode(instructional_query, convert_to_numpy=True)
        return embedding.tolist()