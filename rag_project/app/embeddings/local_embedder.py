import torch
from sentence_transformers import SentenceTransformer
from typing import List

class LocalSolonEmbeddingClient:
    def __init__(self):
        # "mps" tells the Mac to use its Metal GPU (Silicon chip)
        # If MPS isn't available, it defaults to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 Initializing Solon locally on: {self.device}")
        
        self.model = SentenceTransformer(
            "OrdalieTech/Solon-embeddings-large-0.1", 
            device=self.device
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Solon requires a specific instruction for retrieval tasks
        # But for basic feature extraction, simple encoding works:
        embeddings = self.model.encode(texts)
        # Manually convert the resulting NumPy array to a list, otherwise it throws an error..
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        # Solon performs better if you tell it it's a query
        instructional_query = f"query: {query}"
        embedding = self.model.encode(instructional_query)
        return embedding.tolist()