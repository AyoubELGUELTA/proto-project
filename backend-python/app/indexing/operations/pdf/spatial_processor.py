import re
from typing import List, Dict, Any
from app.models.domain import TextUnit

class SpatialProcessor:
    """
    Spécialiste de la corrélation spatiale entre les chunks de texte 
    et les éléments visuels (images/tableaux) du PDF.
    """

    @staticmethod
    def enrich_with_spatial_data(doc, dl_chunks: List[Any]) -> List[TextUnit]:
        enriched_units = []
        used_image_ids = set()
        current_active_headings = [] 
        print(f"TAILLE DE dl_chunks: {len(dl_chunks)}")
        print("\n\n")

        print(dl_chunks)
        print("\n\n")
    

        for i, dl_chunk in enumerate(dl_chunks):
            # 1. Gestion des titres (Héritage contextuel)
            current_active_headings = SpatialProcessor._resolve_headings(dl_chunk, current_active_headings)
            
            # 2. Logique Spatiale (BBox)
            chunk_pages = SpatialProcessor._get_page_numbers(dl_chunk)
            bbox = SpatialProcessor._get_chunk_bbox(dl_chunk)
            
            # 3. Corrélation Images/Tables (La méthode qui manquait !)
            images_found = SpatialProcessor._correlate_images(
                doc, dl_chunk, chunk_pages, bbox, used_image_ids
            )
            
            # 4. Détection de tableaux en Markdown (Optionnel selon ton besoin)
            tables = []
            if bool(re.search(r"\|[- :]+\|", dl_chunk.text or "")):
                tables.append(dl_chunk.text)

            # 5. Création de l'unité
            unit = TextUnit(
                id=f"chunk_{i}",
                text=dl_chunk.text or "",
                headings=current_active_headings, 
                page_numbers=chunk_pages,
                tables=tables,
                metadata={
                    "bbox": bbox,
                    "heading_full": " > ".join(current_active_headings),
                    "docling_images": images_found
                }
            )
            enriched_units.append(unit)
        for j in range (len(enriched_units)):
            print(f" DEBUG LIGNE 50 SPATIAL_PROCESSOR.PY ----- {enriched_units[j].id}")
            print("\n\n")
            print(f"TAILLE DE ENRICHED_UNITS: {len(enriched_units)}")
        return enriched_units

    @staticmethod
    def _correlate_images(doc, dl_chunk, chunk_pages, bbox, used_image_ids) -> List[Any]:
        """Extrait les images à proximité immédiate du chunk de texte."""
        found = []
        c_min, c_max = bbox["top"], bbox["bottom"]

        if not hasattr(doc, 'pictures') or not doc.pictures:
            return found

        for pic in doc.pictures:
            if not pic.prov: continue
            
            pic_prov = pic.prov[0]
            img_id = f"pg_{pic_prov.page_no}_{pic_prov.bbox.l}_{pic_prov.bbox.t}"
            
            if img_id in used_image_ids: continue

            # Vérification de la page et de la proximité verticale
            if pic_prov.page_no in chunk_pages:
                # On tolère une marge de 100 unités pour "coller" l'image au texte
                is_near = (c_min - 100) <= pic_prov.bbox.t <= (c_max + 100)
                is_sole = len(chunk_pages) == 1
                
                if is_near or is_sole:
                    # Filtre anti-doublon pour les tableaux extraits comme images
                    if SpatialProcessor._is_image_a_table_duplicate(pic_prov.bbox, c_min, c_max):
                        continue
                    
                    try:
                        pil_image = pic.get_image(doc)
                        if pil_image:
                            found.append(pil_image)
                            used_image_ids.add(img_id)
                    except Exception as e:
                        print(f"⚠️ Erreur extraction PIL image {img_id}: {e}")
        return found

    @staticmethod
    def _resolve_headings(dl_chunk: Any, previous_headings: List[str]) -> List[str]:
        meta_headings = getattr(dl_chunk.meta, 'headings', []) or []
        if meta_headings:
            return meta_headings
            
        label = getattr(dl_chunk.meta, 'label', '').lower()
        if label in ['heading', 'title']:
            return [dl_chunk.text.strip()]
            
        return previous_headings

    @staticmethod
    def _get_chunk_bbox(dl_chunk: Any) -> Dict[str, float]:
        tops = [item.prov[0].bbox.t for item in dl_chunk.meta.doc_items if item.prov]
        bottoms = [item.prov[0].bbox.b for item in dl_chunk.meta.doc_items if item.prov]
        return {
            "top": min(tops) if tops else 0,
            "bottom": max(bottoms) if bottoms else 1000
        }

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
        pic_height = abs(pic_bbox.b - pic_bbox.t)
        chunk_height = abs(c_bottom - c_top)
        return (pic_height / chunk_height) > 0.05 if chunk_height > 0 else False