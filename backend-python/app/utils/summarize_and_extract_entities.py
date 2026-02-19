import asyncio
from ..db import update_chunks_with_ai_data, link_entity_to_chunk
import os
import json
import re
from typing import List, Dict, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


semaphore = asyncio.Semaphore(10)


async def bounded_process(chunk, chunk_id):
    async with semaphore:
        return await process_single_chunk(chunk, chunk_id)

async def process_single_chunk(chunk_data: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """
    Analyse un chunk pour :
    1. Extraire les faits des images/tableaux (Visual Summary).
    2. Extraire les entités et leurs alias.
    Retourne un dictionnaire complet pour la mise à jour BDD et le linking.
    """
    text = chunk_data.get("text", "")
    tables = chunk_data.get("tables", [])
    images = chunk_data.get("images_urls", [])
    heading = chunk_data.get("heading_full", "Sans titre")
    
    # Flags pour le contexte des tableaux
    is_continuation = chunk_data.get("is_continuation", False)
    is_cut = chunk_data.get("is_cut", False)

    try:
        llm = ChatOpenAI(
            model=os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0, # On veut du déterministe pur
            model_kwargs={"response_format": {"type": "json_object"}} # Force la sortie JSON
        )

        prompt = f"""
        Tu es un expert en extraction d'entités pour un Knowledge Graph islamique.

        CONTEXTE : {heading}
        TEXTE : {text}

        MISSION : Extrais UNIQUEMENT les entités MAJEURES :

        1. PERSONNES : Prophètes, Compagnons, Savants, Mères des Croyants
        - Utilise le nom COMPLET (ex: "Aïcha bint Abi Bakr" pas juste "Aïcha")
        - Inclus honorifiques dans aliases : ["(ra)", "(saw)"]

        2. CONCEPTS RELIGIEUX : Piliers, pratiques, termes techniques
        - Nom arabe + traduction si pertinente
        - Ex: "Salat (Prière)", aliases: ["Salat", "Prière", "Prière rituelle"]

        3. LIEUX : Villes sacrées, lieux historiques
        - Ex: "Médine", aliases: ["Madinah", "Yathrib"]

        4. ÉVÉNEMENTS : Batailles, révélations, migrations
        - Ex: "Hijra", aliases: ["Hégire", "Migration"]

        CRITÈRES DE SÉLECTION :
        - L'entité est mentionnée substantiellement (pas juste en passant)
        - Elle a une importance religieuse/historique
        - IGNORE : pronoms, mots courants, entités secondaires

        THÈMES CONTEXTUELS (pour aide) :
        - Piliers de l'Islam, Jurisprudence, Histoire Prophétique, Spiritualité, etc.

        FORMAT JSON :
        {{
        "visual_summary": "...",
        "entities": [
            {{
            "name": "Aïcha bint Abi Bakr",
            "type": "PERSONNE",
            "aliases": ["Aisha", "Aïcha (ra)", "Mère des Croyants"],
            "relevance": 0.9,
            "themes": ["Mères des Croyants", "Histoire Prophétique"]
            }}
        ]
        }}

        RÈGLES :
        - 3-8 entités MAX par chunk (qualité > quantité)
        - Pas de phrases, JSON pur
        - Si aucune entité majeure : "entities": []
        """

        message_content = [{"type": "text", "text": prompt}]

        # Ajout des tableaux en texte
        if tables:
            tables_text = "\n".join([f"Tableau {i+1}: {t}" for i, t in enumerate(tables)])
            message_content[0]["text"] += f"\n\n--- TABLEAUX ---\n{tables_text}"

        # Ajout des images
        for url in images:
            if isinstance(url, str) and url.startswith("http"):
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"}
                })

        response = await llm.ainvoke([HumanMessage(content=message_content)])
        raw_data = json.loads(cast(str, response.content))
        print (f"RAW_DATA INGEST LLM POUR CHUNK: {chunk_id} : {raw_data}.")
        # Nettoyage du texte original (retrait des tableaux markdown si résumé présent)
        clean_text = text
        if raw_data.get("visual_summary"):
            clean_text = re.sub(r'^\s*\|[\s\-\|]+\|\s*$', '', text, flags=re.MULTILINE)
            clean_text = re.sub(r'^\s*\|.*\|\s*$', '', clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text).strip()

        return {
            "chunk_id": chunk_id,
            "text": clean_text,
            "visual_summary": raw_data.get("visual_summary", ""),
            "entities": raw_data.get("entities", []),
            "heading_full": heading
        }

    except Exception as e:
        print(f"⚠️ Erreur process_single_chunk [{chunk_id}]: {e}")
        return {
            "chunk_id": chunk_id,
            "text": text,
            "visual_summary": "",
            "entities": []
        }
    

async def summarise_and_extract_entities(organized_chunks, chunk_ids):
    # 1. On lance les tâches LLM (Résumé + Entités)
    tasks = [
        bounded_process(chunk, chunk_id) 
        for chunk, chunk_id in zip(organized_chunks, chunk_ids)
    ]
    results = await asyncio.gather(*tasks)

    # results = [{"chunk_id": id, "text": t, "visual_summary": vs, "entities": []}, ...]

    # 2. Mise à jour groupée (Texte + Visual Summary)
    await update_chunks_with_ai_data(results)

    # 3. Linking des entités (Nouveauté)
    linking_tasks = []
    for res in results:
        for entity in res["entities"]:
            linking_tasks.append(link_entity_to_chunk(res["chunk_id"], entity))
    
    if linking_tasks:
        await asyncio.gather(*linking_tasks)
    
    return results