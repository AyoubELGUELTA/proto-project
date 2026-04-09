# app/services/llm/factory.py
from app.services.llm.client import LLMClient
from app.services.llm.service import LLMService # On importe le Service
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import LLMConfig, EXTRACTION_LLM_CONFIG, SUMMARIZATION_LLM_CONFIG
from app.core.settings import settings

class LLMFactory:
    _tracker = LLMTracker()
    _cache = LLMCache(redis_url=settings.redis_url)
    
    @classmethod
    def get_service(cls, config: LLMConfig = None) -> LLMService:
        config = config or EXTRACTION_LLM_CONFIG 
        
        # Logique de création du client...
        client = LLMClient(config=config, api_key=settings.openai_api_key, tracker=cls._tracker, cache=cls._cache)
        
        return LLMService(client=client)

    # Raccourcis sémantiques (très utiles pour la lisibilité)
    @classmethod
    def get_extractor(cls):
        return cls.get_service(EXTRACTION_LLM_CONFIG)

    @classmethod
    def get_summarizer(cls):
        return cls.get_service(SUMMARIZATION_LLM_CONFIG)

    @classmethod
    def get_tracker(cls) -> LLMTracker:
        return cls._tracker