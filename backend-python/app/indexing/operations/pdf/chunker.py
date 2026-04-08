import hashlib
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from app.core.config.ingestion_config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentChunker:
    """Responsable du découpage logique du document."""
    
    def __init__(self):
        self.hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.tokenizer = HuggingFaceTokenizer(tokenizer=self.hf_tokenizer)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=CHUNK_SIZE,
            overlap_tokens=CHUNK_OVERLAP,
            merge_peers=True
        )

    def chunk(self, dl_doc) -> list:
        """Retourne les chunks bruts de Docling."""
        return list(self.chunker.chunk(dl_doc))