import asyncio
import os
import json
import re
from typing import List, Dict, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.core.tags_store import TagsStore
# Importation de tes fonctions DB (assure-toi que les chemins sont corrects)
from ..db import update_chunks_with_ai_data, link_entity_to_chunk

semaphore = asyncio.Semaphore(4)

async def bounded_process(chunk, chunk_id):
    async with semaphore:
        return await process_single_chunk(chunk, chunk_id)

async def process_single_chunk(chunk_data: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """
    Analyse un chunk pour extraire le résumé visuel et les entités 
    en les raccordant à la taxonomie officielle.
    """
    # Récupération de la liste des 78 tags pour le prompt
    tags_context = TagsStore.get_prompt_context()

    text = chunk_data.get("text", "")
    tables = chunk_data.get("tables", [])
    images = chunk_data.get("images_urls", [])
    heading = chunk_data.get("heading_full", "Sans titre")
    
    # Gestion de la continuité
    context_notes = []
    if chunk_data.get("is_continuation"): context_notes.append("- SUITE du chunk précédent.")
    if chunk_data.get("is_cut"): context_notes.append("- COUPÉ, se termine au prochain.")
    if chunk_data.get("is_table_continuation"): context_notes.append("- MILIEU de tableau.")
    
    notes_string = "\n".join(context_notes) if context_notes else "RAS."

    try:
        llm = ChatOpenAI(
            model=os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        prompt = f"""
        Tu es un expert en extraction d'entités et en classification taxonomique pour un Knowledge Graph Islamique.

        TAXONOMIE OFFICIELLE (Tags disponibles) :
        {tags_context}

        CONTEXTE DU DOCUMENT : {heading}
        NOTES DE CONTINUITÉ : {notes_string}
        TEXTE À ANALYSER : {text}

        MISSION :
        1. Résumé Visuel : Si tableaux/images, extrais les faits clés.
        2. Extraction d'Entités : Identifie les Personnes, Lieux, Concepts et Événements.
        3. Classification : Pour CHAQUE entité, choisis le "suggested_tag" le plus précis dans la TAXONOMIE OFFICIELLE ci-dessus.

        RÈGLES CRITIQUES :
        - Entité = Nom Propre (ex: "Aïcha bint Abi Bakr", pas "La femme du prophète").
        - Suggested Tag = Doit correspondre EXACTEMENT à un label de la liste fournie.
        - Si aucun tag précis ne correspond, remonte au parent (ex: "Fiqh (Jurisprudence)").
        - Ne crée JAMAIS de nouveaux tags.

        FORMAT DE SORTIE JSON :
        {{
            "visual_summary": "Texte court",
            "entities": [
                {{
                    "name": "Nom complet",
                    "type": "PERSONNE|LIEU|CONCEPT|EVENEMENT",
                    "aliases": ["Alias1", "Variante Arabe"],
                    "relevance": 0.0-1.0,
                    "suggested_tag": "Nom du tag officiel"
                }}
            ]
        }}
        """

        message_content = [{"type": "text", "text": prompt}]
        if tables:
            message_content[0]["text"] += f"\n\nTABLEAUX :\n{str(tables)}"

        for url in images:
            if isinstance(url, str) and url.startswith("http"):
                message_content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})

        response = await llm.ainvoke([HumanMessage(content=message_content)])
        raw_data = json.loads(cast(str, response.content))
        
        return {
            "chunk_id": chunk_id,
            "text": text,
            "visual_summary": raw_data.get("visual_summary", ""),
            "entities": raw_data.get("entities", []),
            "heading_full": heading
        }

    except Exception as e:
        print(f"⚠️ Erreur LLM [{chunk_id}]: {e}")
        return {"chunk_id": chunk_id, "text": text, "visual_summary": "", "entities": []}

async def summarise_and_extract_entities(organized_chunks, chunk_ids):
    """
    Fonction maître coordonnant l'analyse et le linking.
    """
    # 1. Analyse LLM
    tasks = [bounded_process(chunk, chunk_id) for chunk, chunk_id in zip(organized_chunks, chunk_ids)]
    results = await asyncio.gather(*tasks)

    # 2. Mise à jour des textes et résumés visuels
    await update_chunks_with_ai_data(results)

    # 3. Linking Intelligent (Le Gatekeeper)

    linking_tasks = []
    for res in results:
        for entity in res["entities"]:
            linking_tasks.append(link_entity_to_chunk(res["chunk_id"], entity))
    
    if linking_tasks:
        await asyncio.gather(*linking_tasks)
    
    return results