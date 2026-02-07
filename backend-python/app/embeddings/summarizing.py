import os 
import asyncio
from .create_enhanced_ai_summary import create_ai_enhanced_summary # Assure-toi qu'elle est ASYNC
from ..db.postgres import update_chunks_with_ai_data


semaphore = asyncio.Semaphore(10)

async def process_single_chunk(chunk, chunk_id):
    """
    Traite un chunk individuel avec une file d'attente (Semaphore).
    Conserve la logique originale : si pas de tables/images, on renvoie tel quel.
    """
    text = chunk["text"]
    tables = chunk["tables"]
    images = chunk["images_urls"]
    visual_description = "" 

    if tables or images:
        print(f"🧠 Appel IA (File d'attente) pour chunk {chunk_id} ({len(images)} imgs, {len(tables)} tabs)")
        try:
            # On entre dans la file d'attente ici
            async with semaphore:
                text, visual_description = await create_ai_enhanced_summary(
                    text,
                    tables,
                    images
                )
        except Exception as e:
            print(f"⚠️ Erreur AI summary pour chunk {chunk_id}: {e}")
    
    return {
        "chunk_id": str(chunk_id),
        "text": text,
        "visual_summary": visual_description
    }

async def summarise_chunks(organized_chunks, chunk_ids):
    """
    Résume les chunks en PARALLÈLE. 
    Divise le temps d'ingestion par le nombre de chunks !
    """
    print(f"🚀 Lancement de la synthèse IA en parallèle pour {len(chunk_ids)} chunks...")
    
    # 1. On crée une liste de tâches (coroutines) sans les exécuter encore
    tasks = [
        process_single_chunk(chunk, chunk_id) 
        for chunk, chunk_id in zip(organized_chunks, chunk_ids)
    ]

    # 2. On lance tout en même temps ! 
    # asyncio.gather attend que TOUTES les tâches soient finies
    summarised_chunks = await asyncio.gather(*tasks)

    # 3. Mise à jour groupée dans Postgres
    if summarised_chunks:
        await update_chunks_with_ai_data(summarised_chunks)
        print(f"✅ {len(summarised_chunks)} chunks traités et mis à jour.")
    
    return summarised_chunks