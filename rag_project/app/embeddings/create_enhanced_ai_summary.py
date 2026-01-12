from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from typing import List, Dict, Any, cast
from .normalize_llm_response import normalize_llm_content  # to stringify the llm response

load_dotenv()


def create_ai_enhanced_summary(text: str, tables: list[str], images: list[str]) -> str:
    """
    Merge multimodal content into a compact, embeddable text.
    - If only text is present -> return text unchanged
    - If tables/images are present -> LLM integrates ONLY their factual contribution
    """

    # BYPASS LLM if no multimodal content, economy on credits
    if not tables and not images:
        return text

    try:
        # Initialize LLM (vision-capable for images)
        llm = ChatOpenAI(
            model="gpt-4.1-nano-2025-04-14",
            temperature=0
        )

        # STRICT, NON-EXPANSIVE PROMPT
        prompt_text = f"""
You are processing a document chunk for semantic indexing.

IMPORTANT RULES:
- The TEXT CONTENT is already correct and must NOT be rewritten.
- Do NOT summarize the text.
- Do NOT paraphrase the text.
- Do NOT expand the text.
- Only integrate factual information coming from TABLES and/or IMAGES.
- If tables or images do not add meaningful information, return the text unchanged.

STYLE CONSTRAINTS:
- Keep the output as close as possible to the original text length
- No bullet points unless strictly necessary
- No questions
- No synonym lists
- No SEO wording
- No repetition
- Neutral, factual French
- Maximum length: 900 characters

CONTENT TO PROCESS:

TEXT CONTENT:
{text}
"""

        # Add tables if present
        if tables:
            prompt_text += "\nTABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n"

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
            content=cast(List[Dict[str, Any]], message_content)# type: ignore[arg-type], because everythings work fine
        ) 
        response = llm.invoke([message])

        return normalize_llm_content(response.content)

    except Exception:
        # Fallback: return original text unchanged
        return text
