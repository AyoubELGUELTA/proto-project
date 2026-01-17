import os

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': getattr(chunk, 'text', ''), #fallback si ya pas .text dans chunk
        'tables': [],
        'images_base64': [],
    }

    metadata = getattr(chunk, 'metadata', None)
    if not metadata:
        return content_data
    
    orig_elements = getattr(metadata, 'orig_elements', []) #check si il y a orig_elements ou non
    
    # Debug optionnel pour voir ce qui passe en cas de pépin (configurable via .env)
    debug_mode = os.getenv("DEBUG_INGESTION", "false").lower() == "true"

    for element in orig_elements:
        try:
            element_type = type(element).__name__
            
            # Gestion des Tables
            if element_type == 'Table':
                table_meta = getattr(element, 'metadata', None)
                # On essaie de récupérer le HTML, sinon le texte brut, sinon rien
                table_html = getattr(table_meta, 'text_as_html', getattr(element, 'text', ''))
                if table_html:
                    content_data['tables'].append(table_html)
            
            # Gestion des Images
            elif element_type == 'Image':
                image_meta = getattr(element, 'metadata', None)
                if image_meta:
                    base64_data = getattr(image_meta, 'image_base64', None)
                    if base64_data:
                        content_data['images_base64'].append(base64_data)

        except Exception as e:
            if debug_mode:
                print(f"⚠️ Erreur lors de l'analyse d'un élément du chunk: {e}")
            # On continue la boucle pour ne pas bloquer les autres éléments
            continue
    
    return content_data
