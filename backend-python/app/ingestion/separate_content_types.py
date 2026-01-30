import io
import base64
from docling_core.types.doc import PictureItem, TableItem

def separate_content_types(chunk):
    """
    Analyse un chunk Docling pour extraire le texte, les tables et les images.
    """
    content_data = {
        'text': chunk.text, # Le texte du chunk est déjà en Markdown !
        'tables': [],
        'images_base64': [],
    }

    # On parcourt les éléments sources qui composent ce chunk
    for item in chunk.meta.doc_items:
        
        # 1. Gestion des Tableaux (Docling les a déjà parfaitement convertis)
        if isinstance(item, TableItem):
            # On récupère le Markdown du tableau
            table_md = item.export_to_markdown()
            if table_md not in content_data['tables']:
                content_data['tables'].append(table_md)
        
        # 2. Gestion des Images (Crops haute résolution pour la vision)
        elif isinstance(item, PictureItem):
            try:
                # Docling stocke l'image dans l'attribut .image (objet PIL)
                if hasattr(item, 'image') and item.image:
                    buffered = io.BytesIO()
                    # On garde un format léger pour l'envoi API
                    item.image.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    content_data['images_base64'].append(img_b64)
            except Exception as e:
                print(f"⚠️ Erreur extraction image Docling: {e}")

    return content_data