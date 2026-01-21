import os
import torch
from sentence_transformers import CrossEncoder
import cohere

_LOCAL_MODEL_CACHE = None # This keeps the model in memory to avoid reloading on every request

def get_local_reranker():
    global _LOCAL_MODEL_CACHE
    if _LOCAL_MODEL_CACHE is None:
        # Détection automatique du device (MPS pour ton Mac, CPU pour Docker/Linux)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        print(f"🚀 Loading Local Reranker into memory on: {device}...")
        
        # On peut rendre le modèle local configurable aussi
        model_name = os.getenv("LOCAL_RERANK_MODEL", "mixedbread-ai/mxbai-rerank-xsmall-v1")
        
        _LOCAL_MODEL_CACHE = CrossEncoder(
            model_name, 
            device=device
        )
    return _LOCAL_MODEL_CACHE

MIN_RERANK_SCORE = 0.01 #to test in rEALITY...

def rerank_results(query, retrieved_docs, top_n=8):
    """
    Re-rank the search results based on the current ENVIRONMENT.
    """

    if not retrieved_docs:
        return []
    print(retrieved_docs)
    documents_text = [doc["text"] for doc in retrieved_docs]

    final_results = []
    env = os.getenv("ENVIRONMENT", "development")



    try:
        if env == "development":
            model = get_local_reranker()
            pairs = [[query, text] for text in documents_text]
            scores = model.predict(pairs)

            for i, doc in enumerate(retrieved_docs):
                doc["rerank_score"] = float(scores[i])
                final_results.append(doc)

    # --- PRODUCTION / CLOUD MODE ---
        elif env == "production":

            api_key = os.getenv("COHERE_API_KEY")

            if not api_key:
                print("❌ COHERE_API_KEY manquante. Reranking impossible.")
                return retrieved_docs[:top_n]
            
            co = cohere.Client(api_key)

            response = co.rerank(
                model='rerank-multilingual-v3.0',
                query=query,
                documents=documents_text,
                top_n=len(documents_text)
            )
            
            for result in response.results:
                doc = retrieved_docs[result.index]
                doc["rerank_score"] = result.relevance_score
                final_results.append(doc)

        filtered = [
        r for r in final_results
        if r["rerank_score"] >= MIN_RERANK_SCORE
        ]
        filtered.sort(key=lambda x: x["rerank_score"], reverse=True)

        return filtered[:top_n]
    
    except Exception as e:
        print(f"❌ Erreur critique pendant le reranking: {e}")

        return retrieved_docs[:top_n]