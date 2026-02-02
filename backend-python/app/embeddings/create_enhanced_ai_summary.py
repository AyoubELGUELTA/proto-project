import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from typing import List, Dict, Any, cast
from .normalize_llm_response import normalize_llm_content  # to stringify the llm response

def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """
    Merge multimodal content into a compact, embeddable text.
    - If only text is present -> return text unchanged
    - If tables/images are present -> LLM integrates ONLY their factual contribution
    """

    # BYPASS LLM if no multimodal content, economy on credits
    if not images:
        return text

    try:

        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4.1-nano-2025-04-14")

        # Initialize LLM (vision-capable for images)
        llm = ChatOpenAI(
            model=model_name,
            api_key = api_key,
            temperature=0
        )

        # STRICT, NON-EXPANSIVE PROMPT

        prompt_text = f"""
        Tu es un expert en indexation sémantique.
        
        RÈGLES STRICTES :
        
        - ANALYSE les TABLEAUX en Markdown présents dans le texte : convertis leurs données importantes en phrases descriptives factuelles (ex: "Le tableau indique que la ville de Médine compte X puits...").
        - ANALYSE les IMAGES fournies et intègre UNIQUEMENT leurs infos factuelles manquantes au texte.
        - Génère UNIQUEMENT une description textuelle factuelle des informations qu'ils apportent. 
        - Si les images n'apportent rien de plus, ne renvoie rien.
        - Ne répète pas le texte original, UNIQUEMENT les infos des tableaux et des images.
        - Si aucune image/tableau n'est utile, RENVOIE "RAS".
        
        
        TEXTE DE RÉFÉRENCE (pouvant contenir des tableaux) :
        {text}
        """

        # # Add tables if present
        # if tables:
        #     prompt_text += "\nTABLES:\n"
        #     for i, table in enumerate(tables):
        #         prompt_text += f"Table {i + 1}:\n{table}\n" OBSOLETE, TABLES ARE IN THE TEXT ATTRIBUTE ASWELL

        # Build message content
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt_text}
        ]

        # Add images if present
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        # Invoke LLM
        message = HumanMessage(
            content=message_content
        ) 
        response = llm.invoke([message])
        if "RAS" not in response.content.upper():
            enhanced_content = text + "\n\n[INFO COMPLÉMENTAIRE] : " + response.content
        else:
            enhanced_content = text

        return normalize_llm_content(enhanced_content)

    except Exception as e:
        print(f"⚠️ Erreur LLM Vision: {e}")
        # Fallback: return original text unchanged
        return text
