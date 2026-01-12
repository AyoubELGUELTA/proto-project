import os
import requests
from dotenv import load_dotenv
from typing import List
load_dotenv()
from dotenv import load_dotenv
from pathlib import Path
import os

# Chemin absolu vers le .env
env_path = Path(__file__).parent.parent / ".env"  # ajuste selon ton arborescence
load_dotenv(dotenv_path=env_path)

# Endpoint Inference Provider for Solon embeddings
HF_API_URL = "https://router.huggingface.co/hf-inference/models/OrdalieTech/Solon-embeddings-large-0.1/pipeline/feature-extraction"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}", 
    "Content-Type": "application/json"
}


class SolonEmbeddingClient:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document chunks (feature-extraction).
        """
        payload = {
            "inputs": texts if len(texts) > 1 else texts[0],  # HuggingFace API accepte string ou list[string]
            "options": {"wait_for_model": True}
        }
        print("HF_API_URL:", HF_API_URL)
        print("HF_HEADERS:", HF_HEADERS)
        print("DEBUG payload:", payload)


        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        embeddings = response.json()

        # si un seul texte -> API retourne un vecteur, on wrap en list pour uniformité
        if isinstance(embeddings[0], list):
            return embeddings
        else:
            return [embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed une query unique.
        """
        payload = {
            "inputs": query,
            "options": {"wait_for_model": True}
        }

        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
