import os
import requests
from typing import List

# HF_API_URL = "https://router.huggingface.co/hf-inference/models/OrdalieTech/Solon-embeddings-large-0.1/pipeline/feature-extraction"
# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_HEADERS = {
#     "Authorization": f"Bearer {HF_TOKEN}", 
#     "Content-Type": "application/json"
# }

DEFAULT_API_URL = "https://router.huggingface.co/hf-inference/models/OrdalieTech/Solon-embeddings-large-0.1/pipeline/feature-extraction"


class SolonEmbeddingClient:


    def __init__(self):
        # On récupère les infos depuis l'environnement (injectées par Docker/Override)
        self.api_url = os.getenv("HF_SOLON_URL", DEFAULT_API_URL)
        self.token = os.getenv("HF_TOKEN")
        
        if not self.token:
            # On prévient le développeur si la clé manque au démarrage
            print("⚠️ Warning: HF_TOKEN is not set in environment variables.")

        self.headers = {
            "Authorization": f"Bearer {self.token}", 
            "Content-Type": "application/json"
        }


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document chunks (feature-extraction).
        """
        if not texts:
            return []

        payload = {
            "inputs": texts if len(texts) > 1 else texts[0],  # HuggingFace API accepte string ou list[string]
            "options": {"wait_for_model": True}
        }
        # print("HF_API_URL:", HF_API_URL)
        # print("HF_HEADERS:", HF_HEADERS)
        # print("DEBUG payload:", payload)


        # response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        # response.raise_for_status()
        # embeddings = response.json()

        # # si un seul texte -> API retourne un vecteur, on wrap en list pour uniformité
        # if isinstance(embeddings[0], list):
        #     return embeddings
        # else:
        #     return [embeddings]

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            embeddings = response.json()

            # Normalisation de la réponse (HuggingFace peut être capricieux sur le format)
            if isinstance(embeddings, list) and len(embeddings) > 0:
                if isinstance(embeddings[0], list):
                    return embeddings
                else:
                    return [embeddings]
            return []
            
        except Exception as e:
            print(f"❌ Error during HF Embedding: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed une query unique.
        """
        payload = {
            "inputs": query,
            "options": {"wait_for_model": True}
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ Erreur query embedding: {e}")
 

