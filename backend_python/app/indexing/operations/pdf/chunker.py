from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from app.core.config.ingestion_config import CHUNK_SIZE, CHUNK_OVERLAP

import logging
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Handles the logical decomposition of documents into manageable segments (chunks).
    
    Using Docling's HybridChunker, this class respects the structural hierarchy 
    of the document (headings, paragraphs, tables) while maintaining a strict 
    token limit defined by the embedding model's context window.
    """

    def __init__(self):
        """
        Initializes the chunker with the BGE-M3 tokenizer for precise token counting.
        """ #TODO it is not working, the tokens limit are not respected by the chunker, not a real problem because of the text_splitter though...

        self.hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.tokenizer = HuggingFaceTokenizer(tokenizer=self.hf_tokenizer)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=CHUNK_SIZE,
            overlap_tokens=CHUNK_OVERLAP,
            merge_peers=True
        )

    def chunk(self, dl_doc) -> list:
        """
        Transforms a Docling document into a list of structured chunks.
        
        Args:
            dl_doc: The processed Docling document object.
            
        Returns:
            A list of raw chunks ready for indexing.
        """
        
        logger.info(f"🧩 Chunking document: '{dl_doc.name if hasattr(dl_doc, 'name') else 'Unknown'}'")
        
        try:
            chunks = list(self.chunker.chunk(dl_doc))
            logger.info(f"✅ Created {len(chunks)} chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")
            return chunks
        except Exception as e:
            logger.error(f"❌ Error during chunking process: {e}")
            return []