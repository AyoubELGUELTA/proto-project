import os
from unstructured.chunking.title import chunk_by_title

def create_chunks(elements, chunk_size=None):
    """
    Découpe sémantique par titre. 
    L'intelligence d'Unstructured regroupe les paragraphes sous leurs titres respectifs.
    """   
    
    # 1. Récupération de la taille depuis l'environnement ou paramètre
    if chunk_size is None:
        chunk_size = int(os.getenv("CHUNK_SIZE", 3000))

    # 2. Sécurité : si la liste d'éléments est vide
    if not elements:
        print("⚠️ Aucun élément reçu pour le chunking (PDF vide ou erreur de parsing).")
        return []
    
    try:
        chunks = chunk_by_title(
            elements,
            multipage_sections=True,
            combine_text_under_n_chars=500,
            max_characters=chunk_size,
            new_after_n_chars=2000,
            include_orig_elements=True,
        )
        
        print(f"✅ Chunking réussi : {len(chunks)} chunks créés.")
        return chunks
    
    except Exception as e:
        # En cas d'erreur de chunking, on log l'erreur et on lève une exception claire
        print(f"❌ Erreur critique lors du chunking : {str(e)}")
        # On pourrait retourner elements tel quel, mais il vaut mieux raise 
        # pour éviter d'envoyer des données mal structurées à la suite du pipeline.
        raise RuntimeError(f"Le découpage sémantique a échoué : {e}")