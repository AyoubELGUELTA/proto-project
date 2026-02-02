import io
import base64
from docling_core.types.doc import DoclingDocument, NodeItem, TableItem, PictureItem
from typing import List, Dict, Any, Optional
import re


def separate_content_types(chunk, doc: DoclingDocument):
    """
    Analyse un chunk Docling pour extraire le texte, les tables, les images et les titres.
    
    Args:
        chunk: Le chunk retourné par HybridChunker
        doc: Le document Docling original (nécessaire pour récupérer les images)
    """
    content_data = {
        "chunk_text": chunk.text or "",
        "chunk_headings": [],
        "chunk_heading_full": "",
        "chunk_page_numbers": [],  
        "chunk_tables": [],
        "chunk_images_base64": []
    }

    if hasattr(chunk, 'meta') and chunk.meta:
        if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
            content_data["chunk_headings"] = chunk.meta.headings or []
            content_data["chunk_heading_full"] = " > ".join(chunk.meta.headings) if chunk.meta.headings else ""
        
    content_data["chunk_page_numbers"] = extract_page_numbers(chunk, doc)

    if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'doc_items'):
        for item in chunk.meta.doc_items:
            
            # 3. Gestion des Tableaux
            if isinstance(item, TableItem):
                try:
                    table_md = item.export_to_markdown()
                    if table_md and table_md not in content_data['tables']:
                        content_data['tables'].append(table_md)
                        print(f"  📊 Tableau trouvé et extrait")
                except Exception as e:
                    print(f"⚠️ Erreur extraction tableau: {e}")
            
            # 4. Gestion des Images
            elif isinstance(item, PictureItem):
                try:
                    # Méthode 1: Via l'URI dans doc.pictures
                    if hasattr(item, 'image') and item.image:
                        image_uri = item.image.uri if hasattr(item.image, 'uri') else None
                        
                        if image_uri and hasattr(doc, 'pictures') and image_uri in doc.pictures:
                            pil_image = doc.pictures[image_uri]
                            
                            buffered = io.BytesIO()
                            pil_image.save(buffered, format="JPEG", quality=85)
                            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            content_data['images_base64'].append(img_b64)
                            print(f"  📸 Image trouvée via URI et convertie en base64")
                        
                        # Méthode 2: L'image peut être directement dans item.image (objet PIL)
                        elif hasattr(item.image, 'pil_image'):
                            pil_image = item.image.pil_image
                            
                            buffered = io.BytesIO()
                            pil_image.save(buffered, format="JPEG", quality=85)
                            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            content_data['images_base64'].append(img_b64)
                            print(f"  📸 Image trouvée directement et convertie en base64")
                            
                except Exception as e:
                    print(f"⚠️ Erreur extraction image: {e}")
    
    return content_data



def extract_page_numbers(chunk, doc: DoclingDocument) -> List[int]:
    """
    Extrait tous les numéros de pages couverts par un chunk.
    
    Stratégie:
    1. Parcourir tous les items du chunk
    2. Extraire leur provenance (prov)
    3. Récupérer les page numbers
    """
    page_numbers = set()  # Utiliser un set pour éviter les doublons
    
    # Méthode 1 : Via chunk.meta.doc_items (si disponible)
    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
        for item_ref in chunk.meta.doc_items:
            # item_ref est une référence vers un item du document
            item = doc.get_item(item_ref) if hasattr(doc, 'get_item') else None
            if item and hasattr(item, 'prov'):
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        page_numbers.add(prov.page_no)
    
    # Méthode 2 : Parcourir doc.main_text et vérifier si l'item est dans le chunk
    if not page_numbers:
        for item in doc.main_text:
            if is_item_in_chunk(item, chunk):
                page_no = get_item_page(item)
                if page_no:
                    page_numbers.add(page_no)
    
    # Méthode 3 : Fallback - analyser le texte du chunk pour trouver des patterns
    if not page_numbers:
        # Certains chunks peuvent avoir des infos de page dans leur texte
        # Ex: "Page 15" ou des métadonnées cachées
        chunk_text = chunk.text or ""
        # Pattern pour détecter "page X" dans les métadonnées
        page_matches = re.findall(r'page[:\s]+(\d+)', chunk_text.lower())
        for match in page_matches:
            page_numbers.add(int(match))
    
    # Retourner la liste triée
    return sorted(list(page_numbers))

def get_item_page(item: NodeItem) -> Optional[int]:
    """
    Récupère le numéro de page d'un item Docling.
    """
    if hasattr(item, 'prov') and item.prov:
        for prov in item.prov:
            if hasattr(prov, 'page_no'):
                return prov.page_no
    return None

def is_item_in_chunk(item: NodeItem, chunk) -> bool:
    """
    Vérifie si un item du document fait partie d'un chunk.
    
    Heuristique:
    1. Vérifier si le texte de l'item est dans le texte du chunk
    2. Vérifier les références d'items si disponibles
    """
    # Méthode 1 : Comparaison textuelle
    item_text = getattr(item, 'text', '')
    chunk_text = getattr(chunk, 'text', '')
    
    if item_text and chunk_text and item_text in chunk_text:
        return True
    
    # Méthode 2 : Via les références d'items
    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
        item_id = getattr(item, 'self_ref', None) or id(item)
        chunk_item_ids = [
            getattr(ref, 'self_ref', None) or id(ref) 
            for ref in chunk.meta.doc_items
        ]
        if item_id in chunk_item_ids:
            return True
    
    return False