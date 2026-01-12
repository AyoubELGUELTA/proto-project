import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

HF_API_URL = "https://router.huggingface.co/hf-inference/models/OrdalieTech/Solon-embeddings-large-0.1/pipeline/feature-extraction"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(HF_API_URL, headers=headers, json={"inputs": "Bonjour, test embeddings"})
print(response.status_code)
print(response.text)
