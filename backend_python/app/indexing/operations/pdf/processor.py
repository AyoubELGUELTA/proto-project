import io
import base64
import hashlib
import logging
from app.core.data_model.text_units import TextUnit
from docling_core.types.doc import TableItem, PictureItem

logger = logging.getLogger(__name__)

class ChunkProcessor:
    """
    Refines raw Docling chunks into structured TextUnit domain models.
    
    This processor handles the multimodal extraction layer:
    1. Identity: Generates a stable SHA-256 fingerprint for the text content.
    2. Tables: Converts complex grid structures into LLM-friendly Markdown.
    3. Images: Extracts and encodes visual assets into Base64 for future vision tasks.
    """

    @staticmethod
    def process(chunk, doc) -> TextUnit:
        """
        Analyzes and extracts multimedia data from a structural chunk.
        
        Args:
            chunk: The raw chunk provided by the DocumentChunker.
            doc: The full Docling document (required for image rendering).
            
        Returns:
            A TextUnit object containing text and serialized multimedia assets.
        """
        # 1. Base Extraction
        text = chunk.text or ""
        headings = getattr(chunk.meta, 'headings', [])
        
        # 2. Hash ID (Stability for Caching and Neo4j identity)
        # Using 16 chars to balance uniqueness and database performance
        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        # 3. Multimedia Extraction
        tables = []
        images = []
        
        if hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                if isinstance(item, TableItem):
                    try:
                        md_table = item.export_to_markdown()
                        tables.append(md_table)
                        logger.debug(f"📊 Table extracted in chunk {chunk_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Table export failed in chunk {chunk_id}: {e}")
                
                elif isinstance(item, PictureItem):
                    img_b64 = ChunkProcessor._extract_image(item, doc, chunk_id)
                    if img_b64:
                        images.append(img_b64)

        return TextUnit(
            id=chunk_id,
            text=text,
            headings=headings,
            tables=tables,
            images_b64=images
        )

    @staticmethod
    def _extract_image(item, doc, chunk_id: str) -> str | None:
        """
        Isolated image extraction logic with Base64 encoding.
        
        Converts PIL images to JPEG bytes to optimize storage size 
        before encoding to string.
        """
        try:
            pil_img = item.get_image(doc)
            if pil_img:
                buf = io.BytesIO()
                # Saving as JPEG to reduce the B64 string length compared to PNG
                pil_img.save(buf, format="JPEG", quality=85)
                logger.debug(f"🖼️ Image successfully extracted from chunk {chunk_id}")
                return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.error(f"❌ Failed to render image in chunk {chunk_id}: {e}")
            return None
        return None