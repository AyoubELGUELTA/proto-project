import re
from typing import List, Dict, Any
from app.models.domain import TextUnit

class SpatialProcessor:
    """
    Spécialiste de la corrélation spatiale entre les chunks de texte 
    et les éléments visuels (images/tableaux) du PDF.

    ATTENTION: DEPEND DU FORMAT DES "CHUNKS" (framework), ICI DOCLING.
    """

    @staticmethod
    def enrich_with_spatial_data(doc, dl_chunks: List[Any]) -> List[TextUnit]:
        """
        Parcourt les chunks Docling et leur associe les images/tables 
        en fonction de leur position (BBox).
        """
        enriched_units = []
        used_image_ids = set()

        for i, dl_chunk in enumerate(dl_chunks):
            # 1. Calcul de la zone verticale du chunk (BBox)
            chunk_tops = [item.prov[0].bbox.t for item in dl_chunk.meta.doc_items if item.prov]
            chunk_bottoms = [item.prov[0].bbox.b for item in dl_chunk.meta.doc_items if item.prov]
            
            c_min = min(chunk_tops) if chunk_tops else 0
            c_max = max(chunk_bottoms) if chunk_bottoms else 1000
            
            # Extraction des infos de base via les métadonnées Docling
            chunk_text = dl_chunk.text or ""
            chunk_pages = SpatialProcessor._get_page_numbers(dl_chunk)
            
            # Détection si le texte lui-même contient un tableau Markdown
            tables = []
            if bool(re.search(r"\|[- :]+\|", chunk_text)):
                tables.append(chunk_text)

            # 2. Corrélation avec les images (Pictures) du document
            images_found = []
            if hasattr(doc, 'pictures') and doc.pictures:
                for pic in doc.pictures:
                    if not pic.prov: continue
                    
                    pic_prov = pic.prov[0]
                    img_id = f"pg_{pic_prov.page_no}_{pic_prov.bbox.l}_{pic_prov.bbox.t}"
                    
                    if img_id in used_image_ids: continue

                    # Logique de proximité
                    if pic_prov.page_no in chunk_pages:
                        is_near = (c_min - 100) <= pic_prov.bbox.t <= (c_max + 100)
                        is_sole = len(chunk_pages) == 1
                        
                        if is_near or is_sole:
                            # Filtre anti-doublon (Tableau extrait comme image)
                            if tables and SpatialProcessor._is_image_a_table_duplicate(pic_prov.bbox, c_min, c_max):
                                continue
                            
                            try:
                                pil_image = pic.get_image(doc)
                                if pil_image:
                                    images_found.append(pil_image)
                                    used_image_ids.add(img_id)
                            except Exception as e:
                                print(f"⚠️ Erreur extraction PIL pour image {img_id}: {e}")

            raw_headings = []
            if hasattr(dl_chunk, 'meta') and dl_chunk.meta:
                raw_headings = getattr(dl_chunk.meta, 'headings', []) or []

            unit = TextUnit(
                id=f"chunk_{i}", 
                text=chunk_text,
                headings=raw_headings, 
                page_numbers=chunk_pages,
                tables=tables,
                metadata={
                    "docling_images": images_found,
                    "bbox": {"top": c_min, "bottom": c_max},
                    "heading_full": " > ".join(raw_headings) if raw_headings else ""
                }
            )
            enriched_units.append(unit)

        return enriched_units

    @staticmethod
    def _get_page_numbers(chunk) -> List[int]:
        pages = set()
        if hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                if item.prov:
                    pages.add(item.prov[0].page_no)
        return sorted(list(pages))

    @staticmethod
    def _is_image_a_table_duplicate(pic_bbox, c_top, c_bottom) -> bool:
        """Vérifie si l'image couvre une trop grande partie du chunk (signe d'un tableau)."""
        pic_height = abs(pic_bbox.b - pic_bbox.t)
        chunk_height = abs(c_bottom - c_top)
        if chunk_height > 0:
            return (pic_height / chunk_height) > 0.05
        return False