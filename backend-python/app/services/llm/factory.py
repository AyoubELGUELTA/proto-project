import logging
from app.services.llm.client import LLMClient
from app.services.llm.service import LLMService
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import (
    LLMConfig, 
    LLM_CONFIG_LIGHT, 
    SUMMARIZATION_LLM_CONFIG, 
    LLM_CONFIG_HEAVY
)
from app.core.settings import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Centralized factory for creating and managing LLM services.
    
    It maintains shared instances of the LLMTracker and LLMCache to ensure 
    consistency in token tracking and caching across different service instances.
    """
    
    # Shared instances for the lifecycle of the application
    _tracker = LLMTracker()
    _cache = LLMCache(redis_url=settings.redis_url)
    
    @classmethod
    def get_service(cls, config: LLMConfig = None) -> LLMService:
        """
        Creates a new LLMService instance with a specific configuration.

        Args:
            config (LLMConfig, optional): The model and parameter configuration. 
                                          Defaults to LLM_CONFIG_LIGHT.

        Returns:
            LLMService: A configured service ready for extraction or reasoning.
        """
        config = config or LLM_CONFIG_LIGHT
        
        logger.debug(f"🛠️ Creating LLMService with model: {config.model_name}")
        
        client = LLMClient(
            config=config, 
            api_key=settings.openai_api_key, 
            tracker=cls._tracker, 
            cache=cls._cache
        )
        
        return LLMService(client=client)

    # --- Semantic Shortcuts ---

    @classmethod
    def get_light_extractor(cls) -> LLMService:
        """
        Returns a service configured for fast, cost-effective extractions (e.g., GPT-4o-mini).
        """
        return cls.get_service(LLM_CONFIG_LIGHT)
    
    @classmethod
    def get_heavy_extractor(cls) -> LLMService:
        """
        Returns a service configured for complex reasoning or high-precision extraction (e.g., GPT-4o).
        """
        return cls.get_service(LLM_CONFIG_HEAVY)

    @classmethod
    def get_summarizer(cls) -> LLMService:
        """
        Returns a service specialized in long-context summarization.
        """
        return cls.get_service(SUMMARIZATION_LLM_CONFIG)

    @classmethod
    def get_tracker(cls) -> LLMTracker:
        """
        Provides access to the shared token and cost tracker.
        """
        return cls._tracker