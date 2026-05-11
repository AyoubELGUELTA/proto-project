from typing import Any

def normalize_llm_content(content: Any) -> str:
    """
    Normalize LangChain LLM content (str | list[dict]) into plain text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Common LangChain multimodal patterns
                if "text" in item and isinstance(item["text"], str):
                    texts.append(item["text"])
        return "\n".join(texts)

    # Fallback safety
    return str(content)
