from app.services.llm.client import LLMClient
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import LLMConfig, EXTRACTION_LLM_CONFIG
from app.core.settings import settings

class LLMFactory:
    _tracker = LLMTracker() # Singleton pour toute l'app, permet d'avoir un bilan global de conso
    _cache = LLMCache(redis_url=settings.redis_url) #meme idée pour le cache,     
    
    @classmethod
    def create_client(cls, config: LLMConfig = None) -> LLMClient:
        config = config or EXTRACTION_LLM_CONFIG 
        
        api_key = settings.openai_api_key
        if config.provider == "anthropic":
            api_key = settings.anthropic_api_key # Si je veux tester anthropic plus tard..
            
        return LLMClient(
            config=config,
            api_key=api_key,
            tracker=cls._tracker,
            cache=cls._cache
        )