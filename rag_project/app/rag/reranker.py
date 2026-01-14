import os
from sentence_transformers import CrossEncoder
import cohere

_LOCAL_MODEL_CACHE = None
# This keeps the model in memory to avoid reloading on every request
def get_local_reranker():
    """
    Load the model once and return the cached instance.
    """
    global _LOCAL_MODEL_CACHE
    if _LOCAL_MODEL_CACHE is None:
        print("🚀 Loading Local Reranker into memory (MPS/MacBook Pro)...")
        # Mixedbread is great for French/English retrieval
        _LOCAL_MODEL_CACHE = CrossEncoder(
            'mixedbread-ai/mxbai-rerank-xsmall-v1', 
            device='mps' # Explicitly using Apple Silicon GPU
        )
    return _LOCAL_MODEL_CACHE

def rerank_results(query, retrieved_docs, top_n=8):
    """
    Re-rank the search results based on the current ENVIRONMENT.
    """
    if not retrieved_docs:
        return []

    # Using global environment switch
    env = os.getenv("ENVIRONMENT", "development")
    
    # Extract text from Qdrant hits for the re-ranking models
    documents_text = [doc["metadata"]["original_content"]["raw_text"] for doc in retrieved_docs]

    final_results = []

    # --- DEVELOPMENT / LOCAL MODE ---
    if env == "development":
        # Access the singleton model
        model = get_local_reranker()
        
        # Prepare pairs for the Cross-Encoder
        pairs = [[query, text] for text in documents_text]
        scores = model.predict(pairs)
        
        # Attach scores to the documents
        for i, doc in enumerate(retrieved_docs):
            doc["rerank_score"] = float(scores[i])
            final_results.append(doc)

    # --- PRODUCTION / CLOUD MODE ---
    elif env == "production":
        # API call to Cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        response = co.rerank(
            model='rerank-multilingual-v3.0',
            query=query,
            documents=documents_text,
            top_n=top_n
        )
        
        # Map the results back to our document structure
        for result in response.results:
            doc = retrieved_docs[result.index]
            doc["rerank_score"] = result.relevance_score
            final_results.append(doc)

    # --- POST-PROCESSING ---
    # 1. Sort by the new semantic relevance score
    final_results = sorted(final_results, key=lambda x: x["rerank_score"], reverse=True)

    # 2. Apply threshold filtering
    # Local scores vary, but 0.0 is a safe baseline for 'mxbai'
    threshold = 0.0 if env == "development" else 0.3
    
    # 3. Return the best results limited to top_n
    return [d for d in final_results if d.get("rerank_score", -1) > threshold][:top_n]