from unstructured.chunking.title import chunk_by_title

def create_chunks(elements, chunk_size=3000):
    """
    Découpe sémantique par titre. 
    L'intelligence d'Unstructured regroupe les paragraphes sous leurs titres respectifs.
    """    
    chunks = chunk_by_title(
        elements,
        multipage_sections=True,      # Garde la cohérence même si ça change de page
        combine_text_under_n_chars=500  , # Regroupe les petits paragraphes
        max_characters=chunk_size,    # Limite haute à 2000
        new_after_n_chars=2000,
        include_orig_elements=True,    # pour garder tes tables/images
    )
    return chunks