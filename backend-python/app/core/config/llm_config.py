from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str = "openai" 
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int = 4000

EXTRACTION_LLM_CONFIG = LLMConfig(model_name="gpt-4o-mini", temperature=0.0)
SUMMARIZATION_LLM_CONFIG = LLMConfig(model_name="gpt-4o-mini", temperature=0.1)