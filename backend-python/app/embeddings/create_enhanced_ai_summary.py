import os
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from typing import List, Dict, Any, cast
from .normalize_llm_response import normalize_llm_content  # to stringify the llm response

def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> Tuple[str, str]:
    """
    Merge multimodal content into a compact, embeddable text.
    - If only text is present -> return text unchanged
    - If tables/images are present -> LLM integrates ONLY their factual contribution
    """

    # BYPASS LLM if no multimodal content, economy on credits
    if not images and not tables:
        return text

    try:

        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4.1-nano-2025-04-14")
        # Initialize LLM (vision-capable for images)
        llm = ChatOpenAI(
            model=model_name,
            api_key = api_key,
            temperature=1
        )

        # STRICT, NON-EXPANSIVE PROMPT

        prompt_text  = f"""
        Tu es un extracteur de données factuelles pour indexation sémantique (RAG).
        
        TON OBJECTIF : 
        Extraire les informations complémentaires issues des TABLEAUX et/ou des IMAGES qui ne sont pas explicitement détaillées dans le texte.

        RÈGLES D'OR (STRICTES) :
        1. LOGOS/DÉCO : Si une image est un logo, une icône, une signature ou un élément purement décoratif -> RENVOIE "RAS".
        2. PAS DE BLA-BLA : Ne commence jamais par "L'image montre", "Voici un résumé" ou "Le tableau indique". 
        3. FORMAT DE RÉPONSE : 
        - Produis une liste de faits bruts. Chaque ligne doit être une information autonome.
        - Transforme chaque ligne de tableau en une phrase sujet-verbe-complément.
        - Supprime la structure Markdown (| --- |).
        - NE RECOPIE PAS LE TABLEAU. Décris-le.
        4. TEXTE ORIGINAL : Ne répète jamais ce qui est déjà écrit dans le TEXTE DE RÉFÉRENCE ci-dessous.
        5. SILENCE : Si les visuels n'apportent aucune donnée factuelle supplémentaire -> RENVOIE UNIQUEMENT "RAS".

        STRUCTURE DE SORTIE ATTENDUE :
        - Fait visuel/tabulaire 1
        - Fait visuel/tabulaire 2
        
        TEXTE DE RÉFÉRENCE :
        {text}
        """
        full_prompt = prompt_text

        if tables and len(tables) > 0:
            full_prompt += "\n\n--- TABLEAUX À ANALYSER ---\n"
            for i, table_md in enumerate(tables):
                full_prompt += f"Tableau {i+1}:\n{table_md}\n"
        # Build message content
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": full_prompt}
        ]

        # Add images if present
        for url in images:
            if isinstance(url, str) and url.startswith("http"):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url, 
                        "detail": "low" 
                    }
                })

        # Invoke LLM
        message = HumanMessage(
            content=message_content
        ) 
        response = llm.invoke([message])
        ai_response = response.content.strip()
        if "RAS" not in ai_response.upper():
            print(f"🔍 AI ENRICHMENT LOG : {ai_response}")
            
            # Nettoyage robuste du Markdown pour BGE
            import re
            # 1. Supprime les lignes de séparateurs de tableaux (|---|---|)
            clean_text = re.sub(r'^\s*\|[\s\-\|]+\|\s*$', '', text, flags=re.MULTILINE)
            # 2. Supprime les lignes de contenu (| val |) même si elles sont mal formées
            clean_text = re.sub(r'^\s*\|.*\|\s*$', '', clean_text, flags=re.MULTILINE)
            # 3. Supprime les sauts de ligne multiples créés par le nettoyage
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text).strip()
            
            
            return clean_text, normalize_llm_content(ai_response)
        else:
            return text, ""


    except Exception as e:
        print(f"⚠️ Erreur LLM Vision: {e}")
        # Fallback: return original text unchanged
        return text
