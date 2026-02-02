import os 
from .create_enhanced_ai_summary import create_ai_enhanced_summary

def summarise_chunks(organized_chunks, chunk_ids):
    """
    Résume les chunks et retourne des dicts Python (pas de Document LangChain)
    Plus rapide et plus simple à manipuler.
    """
    summarised_chunks = []

    for chunk, chunk_id in zip(organized_chunks, chunk_ids):
        text = chunk["text"]
        tables = chunk["tables"]
        images = chunk["images_base64"]

        # ✅ Enrichir avec l'IA si tableaux ou images
        if tables or images:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    text,
                    tables,
                    images
                )
            except Exception as e:
                print(f"⚠️ Erreur AI summary pour chunk {chunk_id}: {e}")
                enhanced_content = text
        else:
            enhanced_content = text

        # ✅ Retourner un dict au lieu d'un Document
        summarised_chunk = {
            "chunk_id": str(chunk_id),
            "text": enhanced_content,  # Texte enrichi par l'IA
            "heading_full": chunk["heading_full"],
            "headings": chunk["headings"]
        }

        summarised_chunks.append(summarised_chunk)

    print(f'✅ Premier chunk résumé: {summarised_chunks[0]["chunk_id"]}')
    print(f'   Texte (preview): {summarised_chunks[0]["text"][:150]}...')
    
    return summarised_chunks