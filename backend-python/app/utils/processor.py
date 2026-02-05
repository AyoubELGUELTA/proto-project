from typing import List, Dict, Any
from .s3_storage import storage  
from ..ingestion.separate_content_types import separate_content_types 

def process_enriched_chunks(doc, chunks) -> List[Dict[str, Any]]:
    """
    Transforme les chunks bruts de Docling en chunks enrichis (tables + images uniques).
    """
    uploaded_images = {}
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        # On suppose que separate_content_types est importé ou défini
        content = separate_content_types(chunk, doc)
        
        chunk_images_urls = []
        chunk_pages = content.get("chunk_page_numbers", [])

        # Calcul de la zone verticale du chunk
        chunk_tops = [item.prov[0].bbox.t for item in chunk.meta.doc_items if item.prov]
        chunk_bottoms = [item.prov[0].bbox.b for item in chunk.meta.doc_items if item.prov]
        c_min = min(chunk_tops) if chunk_tops else 0
        c_max = max(chunk_bottoms) if chunk_bottoms else 1000

        if hasattr(doc, 'pictures') and doc.pictures:
            for pic in doc.pictures:
                if not pic.prov: continue
                
                pic_page = pic.prov[0].page_no
                pic_bbox = pic.prov[0].bbox
                img_id = f"pg_{pic_page}_{pic_bbox.l}_{pic_bbox.t}_{pic_bbox.r}_{pic_bbox.b}"

                # Déjà traitée ?
                if img_id in uploaded_images:
                    continue

                if pic_page in chunk_pages:
                    pic_top = pic_bbox.t
                    
                    # Logique de capture robuste
                    is_near = (c_min - 100) <= pic_top <= (c_max + 100)
                    is_sole = len(chunk_pages) == 1 and chunk_pages[0] == pic_page

                    if is_near or is_sole:
                        try:
                            image_obj = pic.get_image(doc)
                            if image_obj:
                                # Filtre taille (200px)
                                if image_obj.size[0] < 200 or image_obj.size[1] < 200:
                                    continue
                                
                                url = storage.upload_image(image_obj)
                                if url:
                                    chunk_images_urls.append(url)
                                    uploaded_images[img_id] = url
                                    print(f"   🎯 IMAGE UNIQUE CAPTURÉE (Page {pic_page}) pour chunk {i}")
                        except Exception as e:
                            print(f"   ⚠️ Erreur extraction pic: {e}")

        enriched_chunks.append({
            'chunk_index': i,
            'text': content['chunk_text'],
            'headings': content['chunk_headings'],
            'heading_full': content.get("chunk_heading_full", "Sans titre"),
            'page_numbers': chunk_pages,
            'tables': content.get('chunk_tables', []),
            'images_urls': chunk_images_urls 
        })
        print(f"📦 Traitement chunk {i+1}/{len(chunks)} terminé.")

    return enriched_chunks