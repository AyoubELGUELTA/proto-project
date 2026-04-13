from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str = "openai" 
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int = 4000
    streaming: bool = False 
    token_report: bool = True

LLM_CONFIG_LIGHT = LLMConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False)
LLM_CONFIG_HEAVY = LLMConfig(model_name="gpt-4o", temperature=0.0, streaming=False)

SUMMARIZATION_LLM_CONFIG = LLMConfig(model_name="gpt-4o-mini", temperature=0.1, streaming=False)

CHAT_AGENT_CONFIG = LLMConfig(model_name="gpt-4o", temperature=0.1, streaming=True)