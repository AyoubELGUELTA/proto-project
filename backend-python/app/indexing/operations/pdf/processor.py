import io
import base64
import hashlib
from app.models.domain import TextUnit
from docling_core.types.doc import TableItem, PictureItem

class ChunkProcessor:
    """Transforme les chunks Docling en modèles TextUnit exploitables."""

    @staticmethod
    def process(chunk, doc) -> TextUnit:
        """Analyse et extrait les données multimédias d'un chunk."""
        
        # 1. Extraction de base
        text = chunk.text or ""
        headings = getattr(chunk.meta, 'headings', [])
        
        # 2. Hash ID (pour le cache et Neo4j)
        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        # 3. Extraction Multimédia
        tables = []
        images = []
        
        if hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                if isinstance(item, TableItem):
                    tables.append(item.export_to_markdown())
                
                elif isinstance(item, PictureItem):
                    img_b64 = ChunkProcessor._extract_image(item, doc)
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
    def _extract_image(item, doc) -> str | None:
        """Logique d'extraction b64 isolée."""
        try:
            # Essai v2 recommandé
            pil_img = item.get_image(doc)
            if pil_img:
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                return base64.b64encode(buf.getvalue()).decode()
        except:
            return None