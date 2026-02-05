import os 
from .create_enhanced_ai_summary import create_ai_enhanced_summary
from ..db.postgres import update_chunks_with_ai_data

async def summarise_chunks(organized_chunks, chunk_ids):
    """
    Résume les chunks et retourne des dicts Python (pas de Document LangChain)
    Plus rapide et plus simple à manipuler.
    """
    summarised_chunks = []

    for chunk, chunk_id in zip(organized_chunks, chunk_ids):
        text = chunk["text"]
        tables = chunk["tables"]
        images = chunk["images_urls"]

        visual_description = "" 

        if tables or images:
            try:
                text, visual_description = create_ai_enhanced_summary(
                    text,
                    tables,
                    images
                )
            except Exception as e:
                print(f"⚠️ Erreur AI summary pour chunk {chunk_id}: {e}")
        
            summarised_chunks.append({
            "chunk_id": str(chunk_id),
            "text": text,
            "visual_summary": visual_description
            })

            
    
    if summarised_chunks:
        await update_chunks_with_ai_data(summarised_chunks)
        
    print(f'✅ Premier chunk résumé: {summarised_chunks[0]["chunk_id"]}')
    print(f'   Texte (preview): {summarised_chunks[0]["text"][:150]}...')
    
    return summarised_chunks