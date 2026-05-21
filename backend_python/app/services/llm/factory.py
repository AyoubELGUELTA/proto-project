import logging
from app.services.llm.client import LLMClient
from app.services.llm.service import LLMService
from app.services.llm.tracker import LLMTracker
from app.services.llm.cache import LLMCache
from app.core.config.llm_config import (
    LLMConfig,
    DOCUMENT_IDENTITY_CONFIG,
    ELEMENT_SUMMARIZATION_CONFIG,
    GRAPH_EXTRACTION_CONFIG,
    ENTITY_RESOLUTION_CONFIG,
    ANCHORING_RESOLUTION_CONFIG,
    CONSULTANT_RESOLUTION_CONFIG,
    COMMUNITY_REPORTING_CONFIG
)
from app.core.settings import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Centralized factory for creating and managing LLM services.
    
    Maintains shared instances of LLMTracker and LLMCache to ensure 
    consistency in token tracking and caching across all task services.
    """
    
    # Shared infrastructure instances for the application lifecycle
    _tracker = LLMTracker()
    _cache = LLMCache(redis_url=settings.redis_url)
    
    @classmethod
    def get_service(cls, config: LLMConfig) -> LLMService:
        """
        Creates a new LLMService instance with a specific target configuration.

        Args:
            config (LLMConfig): The exact model, temperature, and parameter configuration.

        Returns:
            LLMService: A configured service ready to execute its assigned task.
        """
        api_keys = {
            "openai": getattr(settings, "openai_api_key", None),
            "anthropic": getattr(settings, "anthropic_api_key", None),
            "deepseek": getattr(settings, "deepseek_api_key", None),
            "mistral": getattr(settings, "mistral_api_key", None),
            "google": getattr(settings, "google_api_key", None),
        }

        # Sécurité : Vérification préventive de la présence de la clé d'API requise
        provider = config.provider.lower()
        if not api_keys.get(provider):
            logger.warning(
                f"⚠️ API key for provider '{provider}' is missing in settings. "
                f"Requests for task utilizing model '{config.model_name}' might fail."
            )

        client = LLMClient(
            config=config, 
            tracker=cls._tracker, 
            cache=cls._cache,
            api_keys=api_keys
        )
        
        return LLMService(client=client)
    
    # ==============================================================================
    # 🎯 EXPLICIT TASK-BASED SERVICE SHORTCUTS
    # ==============================================================================

    @classmethod
    def get_document_identity_service(cls) -> LLMService:
        """Service tailored for generating structured documents identity profiles."""
        return cls.get_service(DOCUMENT_IDENTITY_CONFIG)
    
    @classmethod
    def get_element_summarization_service(cls) -> LLMService:
        """Service tailored for merging and summarizing entities and relationships text blobs."""
        return cls.get_service(ELEMENT_SUMMARIZATION_CONFIG)
        
    @classmethod
    def get_graph_extraction_service(cls) -> LLMService:
        """Service tailored for high-fidelity tuple extraction from raw text blocks."""
        return cls.get_service(GRAPH_EXTRACTION_CONFIG)

    @classmethod
    def get_entity_resolution_service(cls) -> LLMService:
        """Service tailored for complex cross-entity deduplication and merging operations."""
        return cls.get_service(ENTITY_RESOLUTION_CONFIG)

    @classmethod
    def get_anchoring_resolution_service(cls) -> LLMService:
        """Service tailored for structural context anchoring and verification tasks."""
        return cls.get_service(ANCHORING_RESOLUTION_CONFIG)

    @classmethod
    def get_consultant_resolution_service(cls) -> LLMService:
        """
        Service tailored for semantic entity clustering and historical alias bridging.
        Identifies potential duplicate entities (e.g., matching names with epithets or kunyas)
        and outputs candidate index groupings in a strict JSON list of lists format.
        """
        return cls.get_service(CONSULTANT_RESOLUTION_CONFIG)

    @classmethod
    def get_community_reporting_service(cls) -> LLMService:
        """Service tailored for mass context multi-level community insights and reports generation."""
        return cls.get_service(COMMUNITY_REPORTING_CONFIG)

    # ==============================================================================
    # 📊 MONITORING UTILS
    # ==============================================================================

    @classmethod
    def get_tracker(cls) -> LLMTracker:
        """Provides direct access to the shared session token and cost tracker."""
        return cls._tracker