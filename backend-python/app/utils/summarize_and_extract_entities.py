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
    
    # --- RÉCUPÉRATION DES 4 ATTRIBUTS DE CONTINUITÉ ---
    is_continuation = chunk_data.get("is_continuation", False)
    is_cut = chunk_data.get("is_cut", False)
    is_table_continuation = chunk_data.get("is_table_continuation", False)
    is_table_cut = chunk_data.get("is_table_cut", False)

    # Préparation des notes de contexte pour le LLM
    context_notes = []
    if is_continuation:
        context_notes.append("- ℹ️ Ce passage est la SUITE du chunk précédent.")
    if is_cut:
        context_notes.append("- ℹ️ Ce passage est COUPÉ et se termine dans le chunk suivant.")
    if is_table_continuation:
        context_notes.append("- ⚠️ TABLEAU EN COURS : Tu es au milieu d'un tableau. Ne cherche pas les entêtes, extrais juste les données des lignes présentes.")
    if is_table_cut:
        context_notes.append("- ⚠️ TABLEAU COUPÉ : Ce tableau n'est pas fini, il s'arrête brusquement ici.")

    notes_string = "\n".join(context_notes) if context_notes else "Aucune contrainte de continuité particulière."
    try:
        llm = ChatOpenAI(
            model=os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0, # On veut du déterministe pur
            model_kwargs={"response_format": {"type": "json_object"}} # Force la sortie JSON
        )

        prompt = f"""
        Tu es un expert en extraction d'entités pour un Knowledge Graph.

        CONTEXTE : {heading}

        NOTES DE CONTINUITÉ :
        {notes_string}

        TEXTE : {text}

        MISSION : Extrais les entités importantes (Personnes, Lieux, Concepts religieux, Événements).

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ⚠️ RÈGLES CRITIQUES (s'appliquent à TOUS les types)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        1. NOM PROPRE vs TITRE/RÔLE :
        - Utilise TOUJOURS un nom spécifique comme entité principale
        - Les titres, rôles, et descriptions génériques vont dans aliases
        
        ✅ CORRECT :
        - Personne : "Fatima bint Muhammad" (pas "Fille du Prophète")
        - Lieu : "La Mecque" (pas "Ville sainte")
        - Concept : "Salat" (pas "Pilier de l'Islam")
        - Événement : "Bataille de Badr" (pas "Bataille importante")
        
        ❌ FAUX :
        - "Compagnon du Prophète" ← RÔLE, pas un nom
        - "Lieu sacré" ← DESCRIPTION, pas un nom
        - "Pratique religieuse" ← CATÉGORIE, pas un concept précis

        2. NOMS COMPLETS ET PRÉCIS :
        - Personnes : "Umar ibn al-Khattab" > "Umar"
        - Lieux : "Médine" avec aliases ["Madinah", "Yathrib"]
        - Concepts : "Zakat" avec aliases ["Aumône légale", "Zakât"]
        - Événements : "Hijra" avec aliases ["Hégire", "Migration à Médine"]

        3. UNE ENTITÉ = UN ÉLÉMENT UNIQUE :
        - Si plusieurs personnes/lieux/concepts → crée UNE entité PAR élément
        - "Les compagnons" n'est PAS une entité
        - Mais "Abu Bakr", "Umar", "Uthman" sont 3 entités distinctes

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📋 TYPES D'ENTITÉS À EXTRAIRE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        1. PERSONNES :
        - Nom complet avec filiation (ex: "Aïcha bint Abi Bakr")
        - Honorifiques dans aliases : ["Aïcha (ra)", "Mère des Croyants"]
        - Pertinence élevée si sujet principal du texte

        2. CONCEPTS RELIGIEUX :
        - Termes arabes + traduction si pertinente
        - Ex: "Salat" avec aliases ["Prière", "Prière rituelle", "Ṣalāt"]
        - Piliers, pratiques, termes juridiques, spiritualité
        - NE PAS extraire : mots génériques ("religion", "foi" seuls)

        3. LIEUX :
        - Villes sacrées, sites historiques, régions importantes
        - Ex: "La Mecque" aliases ["Mekka", "Makkah", "Ville sainte"]
        - Inclus variantes orthographiques dans aliases

        4. ÉVÉNEMENTS :
        - Batailles, révélations, migrations, événements historiques majeurs
        - Ex: "Bataille de Uhud" aliases ["Uhud", "Ghazwa Uhud"]
        - Date ou période dans le nom si pertinent

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 CRITÈRES DE PERTINENCE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Score 0.9-1.0 : Sujet PRINCIPAL du chunk (plusieurs paragraphes)
        Score 0.6-0.8 : Mentionné SUBSTANTIELLEMENT (au moins un paragraphe complet)
        Score 0.3-0.5 : Mention BRÈVE mais contextuellement importante
        Score <0.3  : NE PAS EXTRAIRE (mention passagère, exemple simple)

        RÈGLE : Si une entité apparaît juste dans une énumération ou exemple rapide → IGNORE-LA

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🎯 FORMAT DE SORTIE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        {{
        "visual_summary": "Faits extraits des tableaux/images (si présents)",
        "entities": [
            {{
            "name": "Nom Complet et Spécifique",
            "type": "PERSONNE|LIEU|CONCEPT|EVENEMENT",
            "aliases": ["Variante 1", "Traduction", "Forme courte", "Honorifique"],
            "relevance": 0.0-1.0,
            "themes": ["Thème contextuel large"]
            }}
        ]
        }}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ✅ EXEMPLES CORRECTS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Personne :
        {{
        "name": "Umar ibn al-Khattab",
        "type": "PERSONNE",
        "aliases": ["Umar", "Umar (ra)", "Deuxième Calife", "Al-Faruq"],
        "relevance": 0.9,
        "themes": ["Compagnons du Prophète", "Histoire Prophétique"]
        }}

        Concept :
        {{
        "name": "Zakat",
        "type": "CONCEPT",
        "aliases": ["Zakât", "Aumône légale", "Aumône obligatoire", "Pilier de l'Islam"],
        "relevance": 0.8,
        "themes": ["Piliers de l'Islam", "Jurisprudence"]
        }}

        Lieu :
        {{
        "name": "Médine",
        "type": "LIEU",
        "aliases": ["Madinah", "Yathrib", "Ville du Prophète", "Al-Madinah al-Munawwarah"],
        "relevance": 0.7,
        "themes": ["Lieux sacrés", "Histoire Prophétique"]
        }}

        Événement :
        {{
        "name": "Bataille de Badr",
        "type": "EVENEMENT",
        "aliases": ["Badr", "Ghazwa Badr", "Première grande bataille"],
        "relevance": 0.95,
        "themes": ["Batailles prophétiques", "Histoire Prophétique"]
        }}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ❌ EXEMPLES À ÉVITER
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ❌ "Mère des Croyants" → Utilise "Aïcha bint Abi Bakr" (ou autre nom propre)
        ❌ "Ville sainte" → Utilise "La Mecque" ou "Médine"
        ❌ "Pilier de l'Islam" → Utilise "Salat", "Zakat" (concepts précis)
        ❌ "Compagnon" → Utilise "Abu Bakr", "Ali" (noms propres)
        ❌ "Bataille importante" → Utilise "Bataille de Uhud" (nom précis)

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📏 CONTRAINTES FINALES
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        - Extrais 2-8 entités MAX par chunk (qualité > quantité)
        - Ignore les pronoms, mots courants, mentions ultra-brèves
        - Si tableaux/images : extrais les faits NON présents dans le texte
        - Pas de phrases d'introduction, JSON strict uniquement
        - Si aucune entité majeure : {{"entities": []}}
        """

        message_content = [{"type": "text", "text": prompt}]

        # Ajout des tableaux en texte
        if tables:
            tables_text = "\n".join([f"Tableau {i+1}: {t}" for i, t in enumerate(tables)])
            message_content[0]["text"] += f"\n\n--- CONTENU DES TABLEAUX ---\n{tables_text}"

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