
import os
from functools import lru_cache
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer 
from transformers import AutoTokenizer

@lru_cache(maxsize=1)
def get_chunker():
    """Cache le chunker pour éviter de recharger le tokenizer"""
    max_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", 1500))
    
    hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens
    )
    
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)

def create_chunks(doc):
    """
    Découpe le document en respectant la hiérarchie (Layout-Aware).
    """
    chunker = get_chunker()  # Réutilise le chunker en cache
    
    try:
        chunks = list(chunker.chunk(doc))
        print(f"✅ Layout-Aware Chunking réussi : {len(chunks)} chunks créés.")
        # 🔍 Debug : vérifier la taille des chunks
        for i, chunk in enumerate(chunks[:3]):  # 3 premiers chunks
            num_tokens = len(hf_tokenizer.encode(chunk.text))
            num_chars = len(chunk.text)
            print(f"  📄 Chunk {i}: {num_tokens} tokens, {num_chars} caractères")
        return chunks
    except Exception as e:
        print(f"❌ Erreur lors du chunking Docling : {e}")
        raise



def create_chunks(doc):
    """
    Découpe le document en respectant la hiérarchie (Layout-Aware).
    """
    # On récupère la taille depuis l'env (Solon accepte max 512, on laisse une marge pour les résumés de l IA...
    max_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", 430))
    
    # Initialisation du Chunker Intelligent
    chunker = HybridChunker(
        tokenizer="OrdalieTech/Solon-embeddings-large-0.1", 
        max_tokens=max_tokens,
        merge_peers=True  # Regroupe les petits paragraphes de même niveau
    )
    
    try:
        chunks = list(chunker.chunk(doc))
        print(f"✅ Layout-Aware Chunking réussi : {len(chunks)} chunks créés.")
        return chunks
    except Exception as e:
        print(f"❌ Erreur lors du chunking Docling : {e}")
        raise