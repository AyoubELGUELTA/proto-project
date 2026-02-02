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
    chunker = get_chunker()
    
    try:
        chunks = list(chunker.chunk(doc))
        print(f"✅ Layout-Aware Chunking réussi : {len(chunks)} chunks créés.")
        
        # 🔍 Debug : vérifier la taille des chunks
        hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
        for i, chunk in enumerate(chunks[:3]):  # DEBUG 3 premiers chunks
            num_tokens = len(hf_tokenizer.encode(chunk.text))
            num_chars = len(chunk.text)
            print(f"  📄 Chunk {i}: {num_tokens} tokens, {num_chars} caractères")
        
        return chunks
    except Exception as e:
        print(f"❌ Erreur lors du chunking Docling : {e}")
        raise