import re
from typing import List, Dict, Any
from app.core.data_model.text_units import TextUnit

import logging
logger = logging.getLogger(__name__)

class SpatialProcessor:
    """
    Expert in spatial correlation between text chunks and visual elements.
    
    This processor reconstructs the document's context by:
    1. Propagating hierarchical headings (Contextual Heritage).
    2. Analyzing Bounding Boxes (BBox) to anchor images and tables to text.
    3. Filtering visual redundancies to ensure clean data for the LLM.
    """

    @staticmethod
    def enrich_with_spatial_data(doc, dl_chunks: List[Any]) -> List[TextUnit]:
        """
        Enriches raw chunks with spatial metadata and correlated visual assets.
        
        Args:
            doc: The full Docling document object.
            dl_chunks: List of raw chunks from the DocumentChunker.
            
        Returns:
            List[TextUnit]: Units enriched with headings, page numbers, and linked images.
        """
        logger.info(f"🌐 Starting spatial enrichment for {len(dl_chunks)} chunks...")
        enriched_units = []
        used_image_ids = set()
        current_active_headings = [] 

        for i, dl_chunk in enumerate(dl_chunks):
            # 1. Heading Management (Inheritance)
            current_active_headings = SpatialProcessor._resolve_headings(dl_chunk, current_active_headings)
            
            # 2. Spatial Logic (BBox & Pages)
            chunk_pages = SpatialProcessor._get_page_numbers(dl_chunk)
            bbox = SpatialProcessor._get_chunk_bbox(dl_chunk)
            
            # 3. Image/Table Correlation
            images_found = SpatialProcessor._correlate_images(
                doc, dl_chunk, chunk_pages, bbox, used_image_ids
            )
            
            # 4. Table Detection (Markdown pattern matching)
            tables = []
            if dl_chunk.text and bool(re.search(r"\|[- :]+\|", dl_chunk.text)):
                tables.append(dl_chunk.text)
                logger.debug(f"📊 Markdown table detected in chunk {i}")

            # 5. Unit Assembly
            unit = TextUnit(
                id=f"chunk_{i}",
                text=dl_chunk.text or "",
                headings=current_active_headings, 
                page_numbers=chunk_pages,
                tables=tables,
                metadata={
                    "bbox": bbox,
                    "heading_full": " > ".join(current_active_headings),
                    "docling_images": [True for _ in images_found] # Flag for presence
                }
            )
            # Link PIL images to unit if your TextUnit model supports it, 
            # otherwise encode to B64 here.
            enriched_units.append(unit)
        
        logger.info(f"✅ Spatial enrichment complete. {len(enriched_units)} units finalized.")
        return enriched_units

    @staticmethod
    def _correlate_images(doc, dl_chunk, chunk_pages, bbox, used_image_ids) -> List[Any]:
        """
        Extracts images located in immediate vertical proximity to a text chunk.
        
        Uses a 'proximity window' strategy: images within 100 units of the text's
        vertical boundaries on the same page are considered contextually linked.
        """
        found = []
        c_min, c_max = bbox["top"], bbox["bottom"]

        if not hasattr(doc, 'pictures') or not doc.pictures:
            return found

        for pic in doc.pictures:
            if not pic.prov: continue
            
            pic_prov = pic.prov[0]
            # Unique ID based on page and coordinates to avoid duplicate extraction
            img_id = f"pg_{pic_prov.page_no}_{pic_prov.bbox.l}_{pic_prov.bbox.t}"
            
            if img_id in used_image_ids: continue

            if pic_prov.page_no in chunk_pages:
                # Tolerance of 100 units to 'glue' the image to its caption/text
                is_near = (c_min - 100) <= pic_prov.bbox.t <= (c_max + 100)
                is_sole = len(chunk_pages) == 1 # If chunk covers full page
                
                if is_near or is_sole:
                    if SpatialProcessor._is_image_a_table_duplicate(pic_prov.bbox, c_min, c_max):
                        continue
                    
                    try:
                        pil_image = pic.get_image(doc)
                        if pil_image:
                            found.append(pil_image)
                            used_image_ids.add(img_id)
                            logger.debug(f"🖼️ Linked image {img_id} to text via spatial proximity.")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to render correlated image {img_id}: {e}")
        return found

    @staticmethod
    def _resolve_headings(dl_chunk: Any, previous_headings: List[str]) -> List[str]:
        """
        Determines the current heading context for a chunk.
        
        If the chunk contains its own heading metadata, it updates the context.
        Otherwise, it inherits the context from the previous chunk to ensure
        semantic continuity (Contextual Heritage).
        """
        meta_headings = getattr(dl_chunk.meta, 'headings', []) or []
        if meta_headings:
            return meta_headings
            
        label = getattr(dl_chunk.meta, 'label', '').lower()
        if label in ['heading', 'title']:
            return [dl_chunk.text.strip()]
            
        return previous_headings

    @staticmethod
    def _get_chunk_bbox(dl_chunk: Any) -> Dict[str, float]:
        """
        Computes the global bounding box of a chunk by aggregating its items.
        
        Returns the 'top' and 'bottom' coordinates to allow vertical 
        proximity checks.
        """
        tops = [item.prov[0].bbox.t for item in dl_chunk.meta.doc_items if item.prov]
        bottoms = [item.prov[0].bbox.b for item in dl_chunk.meta.doc_items if item.prov]
        return {
            "top": min(tops) if tops else 0,
            "bottom": max(bottoms) if bottoms else 1000
        }

    @staticmethod
    def _get_page_numbers(chunk) -> List[int]:
        """
        Identifies all physical pages covered by a single logical chunk.
        """
        pages = set()
        if hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                if item.prov:
                    pages.add(item.prov[0].page_no)
        return sorted(list(pages))

    @staticmethod
    def _is_image_a_table_duplicate(pic_bbox, c_top, c_bottom) -> bool:
        """
        Heuristic to avoid double-extracting tables as both text and images.
        
        If an image's height is very similar to the chunk's height, it's 
        often a graphical table already parsed as text/markdown.
        """
        pic_height = abs(pic_bbox.b - pic_bbox.t)
        chunk_height = abs(c_bottom - c_top)
        # Ratio-based detection: images that 'perfectly' fit the text block
        return (pic_height / chunk_height) > 0.05 if chunk_height > 0 else False