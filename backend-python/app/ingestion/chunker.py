import os
from functools import lru_cache
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer 
from transformers import AutoTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownPictureSerializer
import base64
import io

@lru_cache(maxsize=1)

# Configuration du Serializer spécial pour les images
class ImgAnnotationSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=MarkdownPictureSerializer(),
        )
    
def get_chunker():
    """Cache le chunker pour éviter de recharger le tokenizer"""
    max_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", 1500))
    hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
    )
    
    # C'est ICI qu'on injecte le SerializerProvider
    return HybridChunker(
        tokenizer=tokenizer, 
        merge_peers=True,
        serializer_provider=ImgAnnotationSerializerProvider(),
        max_tokens = max_tokens
    )

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

def extract_single_image_base64(item, doc):
    """Extrait les pixels d'un PictureItem et les convertit en Base64."""
    try:
        # Tente de récupérer l'image PIL directement
        image_obj = item.get_image(doc) # Méthode recommandée v2
        if image_obj:
            buffered = io.BytesIO()
            image_obj.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"      ⚠️ Erreur extraction image: {e}")
    return None