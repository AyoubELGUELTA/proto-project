# app/services/llm/factory.py
from app.services.llm.client import LLMClient
from app.services.llm.service import LLMService # On importe le Service
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import LLMConfig, LLM_CONFIG_LIGHT, SUMMARIZATION_LLM_CONFIG, LLM_CONFIG_HEAVY
from app.core.settings import settings

class LLMFactory:
    _tracker = LLMTracker()
    _cache = LLMCache(redis_url=settings.redis_url)
    
    @classmethod
    def get_service(cls, config: LLMConfig = None) -> LLMService:
        config = config or LLM_CONFIG_LIGHT #Default value
        
        # Logique de création du client...
        client = LLMClient(config=config, api_key=settings.openai_api_key, tracker=cls._tracker, cache=cls._cache)
        
        return LLMService(client=client)

    # Raccourcis sémantiques (très utiles pour la lisibilité)
    @classmethod
    def get_light_extractor(cls):
        return cls.get_service(LLM_CONFIG_LIGHT)
    
    @classmethod
    def get_heavy_extractor(cls):
        return cls.get_service(LLM_CONFIG_HEAVY)

    @classmethod
    def get_summarizer(cls):
        return cls.get_service(SUMMARIZATION_LLM_CONFIG)

    @classmethod
    def get_tracker(cls) -> LLMTracker:
        return cls._tracker