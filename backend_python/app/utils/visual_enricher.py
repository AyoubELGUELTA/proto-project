import asyncio
import os
import json
from typing import List, Dict, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from app.db import update_chunks_with_ai_data

semaphore = asyncio.Semaphore(8) #Between 5 and 10 is fine

async def process_single_chunk_visual(chunk_data: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """
    Ne traite QUE le résumé visuel si nécessaire.
    """
    text = chunk_data.get("text", "")
    tables = chunk_data.get("tables", [])
    images = chunk_data.get("images_urls", [])
    heading = chunk_data.get("heading_full", "Sans titre")

    # --- ÉTAPE 1 : LE COURT-CIRCUIT (Optimisation Coût) ---
    if not tables and not images:
        return {
            "chunk_id": chunk_id,
            "text": text,
            "visual_summary": "", # Rien à ajouter
            "heading_full": heading
        }

    # --- ÉTAPE 2 : APPEL LLM FOCALISÉ ---
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        prompt = f"""
        Tu es un assistant chargé d'indexer des documents historiques (Sira).
        CONTEXTE DU TEXTE : {heading}

        MISSION : 
        Analyse les éléments visuels (tableaux/images) fournis et génère un résumé TECHNIQUE et CONCIS. 
        Ne décris pas la forme, concentre-toi sur le FOND : mots-clés, noms cités, thématiques principales.
        
        RÈGLE : Maximum 3 phrases ou une liste de mots-clés.
        
        FORMAT DE SORTIE JSON :
        {{
            "visual_summary": "Sujet du tableau : [Mots-clés / Idées]"
        }}
        """

        message_content = [{"type": "text", "text": prompt}]
        if tables:
            message_content[0]["text"] += f"\n\nCONTENU DES TABLEAUX :\n{str(tables)}"

        for url in images:
            if isinstance(url, str) and url.startswith("http"):
                message_content.append({"type": "image_url", "image_url": {"url": url, "detail": "low"}})

        response = await llm.ainvoke([HumanMessage(content=message_content)])
        raw_data = json.loads(cast(str, response.content))
        
        return {
            "chunk_id": chunk_id,
            "text": text,
            "visual_summary": raw_data.get("visual_summary", ""),
            "heading_full": heading
        }

    except Exception as e:
        print(f"⚠️ Erreur Visual LLM [{chunk_id}]: {e}")
        return {"chunk_id": chunk_id, "text": text, "visual_summary": "", "heading_full": heading}

async def enrich_chunks_with_visuals(organized_chunks: List[Dict], chunk_ids: List[str]):
    """
    Fonction maître pour le flux vectoriel (Postgres/Qdrant).
    """
    tasks = [process_single_chunk_visual(chunk, chunk_id) for chunk, chunk_id in zip(organized_chunks, chunk_ids)]
    results = await asyncio.gather(*tasks)

    # Mise à jour des métadonnées dans Postgres
    await update_chunks_with_ai_data(results)
    
    return results