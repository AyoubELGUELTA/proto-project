from unstructured.chunking.basic import chunk_elements

def create_chunks(elements, chunk_size=500, overlap=100):
    """
    Découpe les éléments atomiques en chunks intelligents :
    - Texte : découpe en utilisant sauts de ligne ou points pour ne pas couper les phrases
    - Tables / Images : un chunk par élément
    """    
    chunks = chunk_elements(
        elements, # The parsed PDF elements from previous step
        include_orig_elements = True, # we need the original elements
        max_characters=800, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=chunk_size, # Try to start a new chunk after 2400 characters
        overlap=overlap,

    )

    # Stocke les chunks dans Postgres
    return chunks
    

