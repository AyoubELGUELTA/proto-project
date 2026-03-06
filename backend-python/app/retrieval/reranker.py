import os
import torch
from sentence_transformers import CrossEncoder
import cohere
import time 

_LOCAL_MODEL_CACHE = None
_DEVICE_VERIFIED = False
_USE_COHERE_FALLBACK = False

def verify_gpu_performance():
    """
    Teste si le GPU est vraiment utilisé et performant.
    Retourne True si GPU OK, False si fallback Cohere recommandé.
    """
    global _DEVICE_VERIFIED, _USE_COHERE_FALLBACK
    
    if _DEVICE_VERIFIED:
        return not _USE_COHERE_FALLBACK
    
    # Détection device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️ Aucun GPU détecté, passage automatique à Cohere")
        _USE_COHERE_FALLBACK = True
        _DEVICE_VERIFIED = True
        return False
    
    # Test performance GPU
    print(f"🧪 Test performance GPU ({device})...")
    
    try:
        model = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            device=device,
            max_length=512
        )
        
        # Vérifie device réel du modèle
        actual_device = str(next(model.model.parameters()).device)
        print(f"   📍 Modèle chargé sur : {actual_device}")
        
        if "cpu" in actual_device.lower():
            print(f"   ❌ Modèle sur CPU malgré {device} disponible")
            _USE_COHERE_FALLBACK = True
            _DEVICE_VERIFIED = True
            return False
        
        # Benchmark vitesse
        query = "Test performance"
        docs = ["Document " + str(i) for i in range(50)]
        pairs = [[query, doc] for doc in docs]
        
        start = time.perf_counter()
        _ = model.predict(pairs)
        duration = time.perf_counter() - start
        
        print(f"   ⏱️ Benchmark : {duration:.2f}s pour 50 chunks")
        
        # Si >10s pour 50 chunks → trop lent, fallback Cohere
        if duration > 10:
            print(f"   ❌ GPU trop lent ({duration:.2f}s), passage à Cohere")
            _USE_COHERE_FALLBACK = True
        else:
            print(f"   ✅ GPU performant, utilisation locale")
            _USE_COHERE_FALLBACK = False
        
        _DEVICE_VERIFIED = True
        return not _USE_COHERE_FALLBACK
        
    except Exception as e:
        print(f"   ❌ Erreur test GPU : {e}")
        print(f"   → Fallback Cohere activé")
        _USE_COHERE_FALLBACK = True
        _DEVICE_VERIFIED = True
        return False


def get_local_reranker():
    """Charge le reranker local (appelé seulement si GPU OK)."""
    global _LOCAL_MODEL_CACHE
    
    if _LOCAL_MODEL_CACHE is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"🚀 Loading Local Reranker on: {device}...")
        
        model_name = os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        
        _LOCAL_MODEL_CACHE = CrossEncoder(
            model_name, 
            device=device,
            max_length=512
        )
    
    return _LOCAL_MODEL_CACHE


def rerank_with_cohere(query, retrieved_docs, top_n=15):
    """Reranking via Cohere API."""
    api_key = os.getenv("COHERE_API_KEY")
    
    if not api_key:
        print("❌ COHERE_API_KEY manquante, retour docs non-reranked")
        return retrieved_docs[:top_n]
    
    documents_text = [
        doc.get("text_for_reranker", doc.get("text", "")) 
        for doc in retrieved_docs
    ]
    
    try:
        co = cohere.Client(api_key)
        
        response = co.rerank(
            model='rerank-multilingual-v3.0',
            query=query,
            documents=documents_text,
            top_n=min(top_n, len(documents_text))
        )
        
        final_results = []
        for result in response.results:
            doc = retrieved_docs[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            final_results.append(doc)
        
        return final_results
        
    except Exception as e:
        print(f"❌ Erreur Cohere reranking: {e}")
        return retrieved_docs[:top_n]


def rerank_with_local(query, retrieved_docs, top_n=15):
    """Reranking local (GPU)."""
    model = get_local_reranker()
    
    documents_text = [
        doc.get("text_for_reranker", doc.get("text", "")) 
        for doc in retrieved_docs
    ]
    
    pairs = [[query, text] for text in documents_text]
    scores = model.predict(pairs)
    
    final_results = []
    for i, doc in enumerate(retrieved_docs):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = float(scores[i])
        final_results.append(doc_copy)
    
    final_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return final_results[:top_n]


MIN_RERANK_SCORE = 0

def rerank_results(query, retrieved_docs, top_n=15):
    """
    Re-rank avec détection automatique GPU → Cohere fallback.
    """
    start_rerank = time.perf_counter()
    
    if not retrieved_docs:
        return []
    
    # Vérifie GPU performance (une seule fois au démarrage)
    gpu_ok = verify_gpu_performance()
    
    # Choix stratégie
    if gpu_ok:
        print(f"🔄 Reranking LOCAL (GPU) : {len(retrieved_docs)} chunks")
        final_results = rerank_with_local(query, retrieved_docs, top_n)
    else:
        print(f"🌐 Reranking COHERE (API) : {len(retrieved_docs)} chunks")
        final_results = rerank_with_cohere(query, retrieved_docs, top_n)
    
    # Filtrage score minimum
    filtered = [r for r in final_results if r["rerank_score"] >= MIN_RERANK_SCORE]
    
    duration = time.perf_counter() - start_rerank
    print(f"⏱️ RERANKING DURATION: {duration:.2f}s")
    
    return filtered[:top_n]