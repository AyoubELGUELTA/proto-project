def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': getattr(chunk, 'text', ''), #fallback si ya pas .text dans chunk
        'tables': [],
        'images_base64': [],
    }
    
    orig_elements = getattr(getattr(chunk, 'metadata', None), 'orig_elements', []) #check si il y a orig_elements ou non
    
    for element in orig_elements:
        element_type = type(element).__name__#on take le type de chaque éléments du chunk
        
        # Tables
        if element_type == 'Table':
            table_html = getattr(getattr(element, 'metadata', None), 'text_as_html', getattr(element, 'text', ''))
            if table_html:
                content_data['tables'].append(table_html)
        
        # Images
        elif element_type == 'Image':
            image_meta = getattr(element, 'metadata', None)
            if image_meta:
                if getattr(image_meta, 'image_base64', None):
                    content_data['images_base64'].append(image_meta.image_base64)
    
    return content_data
